"""Persistent archive of preprocessing output.

Prediction and HMM search cost ~0.6 s/genome; everything downstream costs
milliseconds. A cold FastAAI run is >98% preprocessing, so it must be paid once
and stored — otherwise the counting engine's speed is irrelevant, and any change
to filter semantics or model set means half an hour of recompute.

**Proteins and raw hits are stored, not the surviving SCPs.** Storing only the
SCPs would weld the archive to one model set. Keeping the full output decouples
all three stages:

    prodigal  ~0.6 s/genome  ->  proteins.faa.gz   (reusable across model sets)
    hmmsearch ~expensive     ->  hits.tsv.gz       (reusable across filter modes)
    resolve   ~free          ->  SCPs
    k-merise  ~free          ->  index

So a different SCP set — GTDB's, say — needs only a re-search of stored proteins,
and a change of filter semantics needs no search at all.

Layout::

    archive/
      proteins/<genome>.faa.gz     standard FASTA, readable by any tool
      hits.tsv.gz                  genome, protein, accession, score
      manifest.tsv                 genome, translation_table, n_proteins, n_hits
      models.txt                   accession list, in order, as searched
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Iterable, Iterator

from .search import Hit

PROTEIN_DIR = "proteins"
HITS_FILE = "hits.tsv.gz"
MANIFEST_FILE = "manifest.tsv"
MODELS_FILE = "models.txt"


def _safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


class Archive:
    """Streaming writer. Genomes are appended as they finish preprocessing."""

    def __init__(self, root, accessions: Iterable[str]):
        self.root = Path(root)
        (self.root / PROTEIN_DIR).mkdir(parents=True, exist_ok=True)
        (self.root / MODELS_FILE).write_text("\n".join(accessions) + "\n")
        self._hits = gzip.open(self.root / HITS_FILE, "wt")
        self._hits.write("genome\tprotein\taccession\tscore\n")
        self._man = open(self.root / MANIFEST_FILE, "w")
        self._man.write("genome\ttranslation_table\tn_proteins\tn_hits\n")
        self._n = 0

    def add(
        self,
        genome: str,
        proteins: dict[str, str],
        hits: list[Hit],
        translation_table: int | None,
    ) -> None:
        path = self.root / PROTEIN_DIR / f"{_safe(genome)}.faa.gz"
        with gzip.open(path, "wt") as fh:
            for name, seq in proteins.items():
                fh.write(f">{name}\n{seq}\n")
        for h in hits:
            self._hits.write(f"{genome}\t{h.protein}\t{h.accession}\t{h.score:.4f}\n")
        self._man.write(
            f"{genome}\t{translation_table if translation_table is not None else 'meta'}"
            f"\t{len(proteins)}\t{len(hits)}\n"
        )
        self._n += 1

    def close(self) -> int:
        self._hits.close()
        self._man.close()
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_models(root) -> list[str]:
    return (Path(root) / MODELS_FILE).read_text().split()


def read_proteins(root, genome: str) -> dict[str, str]:
    """Proteins for one genome, straight back from the stored FASTA."""
    path = Path(root) / PROTEIN_DIR / f"{_safe(genome)}.faa.gz"
    out: dict[str, str] = {}
    name, buf = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def read_hits(root) -> Iterator[tuple[str, list[Hit]]]:
    """Yield `(genome, hits)` grouped in file order."""
    path = Path(root) / HITS_FILE
    current, batch = None, []
    with gzip.open(path, "rt") as fh:
        next(fh)  # header
        for line in fh:
            g, prot, acc, score = line.rstrip("\n").split("\t")
            if g != current:
                if current is not None:
                    yield current, batch
                current, batch = g, []
            batch.append(Hit(prot, acc, float(score)))
    if current is not None:
        yield current, batch


def genome_names(root) -> list[str]:
    """Genome order as recorded in the manifest — the canonical output order."""
    lines = (Path(root) / MANIFEST_FILE).read_text().splitlines()[1:]
    return [ln.split("\t")[0] for ln in lines if ln]


def size_report(root) -> dict[str, float]:
    root = Path(root)
    prot = sum(f.stat().st_size for f in (root / PROTEIN_DIR).glob("*.faa.gz"))
    hits = (root / HITS_FILE).stat().st_size if (root / HITS_FILE).exists() else 0
    return {"proteins_mb": prot / 1e6, "hits_mb": hits / 1e6,
            "total_mb": (prot + hits) / 1e6}
