"""The first two ranks of preprocessing output: gene calls and raw HMM hits.

Prediction and HMM search cost ~2.8 s/genome; everything downstream costs
milliseconds. A cold FastAAI run is >98% preprocessing, so it is written down as
it happens rather than recomputed — otherwise the counting engine's speed is
irrelevant, and any change to filter semantics or model set means hours of
recompute.

**Proteins and raw hits are stored, not the surviving SCPs.** Storing only the
SCPs would weld this rank to one model set. Keeping the full output decouples
all three stages::

    prodigal   ~1.79 s/genome  ->  proteins/<genome>.fasta   reusable across model sets
    hmmsearch  ~1.03 s/genome  ->  hmm_hits/<genome>.tsv     reusable across filter modes
    resolve    ~free           ->  crystals/                 see `crystal.py`
    k-merise   ~free           ->  database/

So a different SCP set — GTDB's, say — needs only a re-search of stored
proteins, and a change of filter semantics needs no search at all.

**One file per genome, in both ranks.** The hits were once a single
`hits.tsv.gz` covering the whole collection, which made re-running one genome a
rewrite of the collection and made the rank impossible to subset by copying
files. Standard formats, one genome each, listable and deletable individually.

Layout and compression are `layout.py`'s: files are plain text unless `--gzip`
was given, and readers accept either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from . import layout
from .search import Hit

MODELS_FILE = "models.txt"
FINGERPRINT_FILE = "models.sha256"

HITS_COLUMNS = "protein\taccession\tscore\n"


def _name_line(genome: str) -> str:
    """The genome's true name, as a comment above the table.

    A filename is not the name: `layout.safe` rewrites anything outside
    `[alnum]._-`, so recovering names from the directory listing would quietly
    rename such a genome, and that altered name would become its identity in
    every database built from this. One comment line per file makes each file
    say what it is, the same way a crystal record does.
    """
    from urllib.parse import quote

    return f"# genome={quote(genome, safe='._-~')}\n"


def _read_name_line(line: str) -> str | None:
    from urllib.parse import unquote

    if line.startswith("# genome="):
        return unquote(line[len("# genome="):].strip())
    return None


class Archive:
    """Writer for the protein and hit ranks. Genomes are written as they finish.

    Not a streaming *file* writer — there is no open handle spanning the run,
    because each genome is a complete file of its own. That is what lets workers
    write concurrently and lets a run be resumed or extended a genome at a time.
    """

    def __init__(self, root, accessions: Iterable[str], fingerprint: str = "",
                 compress: bool = False):
        self.root = Path(root)
        self.compress = compress
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / layout.PROTEINS).mkdir(parents=True, exist_ok=True)
        (self.root / layout.HMM_HITS).mkdir(parents=True, exist_ok=True)
        # Provenance, not data: two small files naming the models these hits
        # were made with, so a rebuild can say what produced them.
        (self.root / MODELS_FILE).write_text("\n".join(accessions) + "\n")
        if fingerprint:
            (self.root / FINGERPRINT_FILE).write_text(fingerprint + "\n")
        self._n = 0

    def add(self, genome: str, proteins: dict[str, str], hits: list[Hit],
            translation_table: int | None = None) -> None:
        stem = layout.safe(genome)
        layout.write_text(
            self.root / layout.PROTEINS / f"{stem}{layout.FASTA_EXT}",
            "".join(f">{name}\n{seq}\n" for name, seq in proteins.items()),
            self.compress,
        )
        layout.write_text(
            self.root / layout.HMM_HITS / f"{stem}{layout.TABLE_EXT}",
            _name_line(genome) + HITS_COLUMNS + "".join(
                f"{h.protein}\t{h.accession}\t{h.score:.4f}\n" for h in hits),
            self.compress,
        )
        self._n += 1

    def close(self) -> int:
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_models(root) -> list[str]:
    return (Path(root) / MODELS_FILE).read_text().split()


def read_fingerprint(root) -> str:
    """Model-set digest, or empty for output written before it was stored."""
    path = Path(root) / FINGERPRINT_FILE
    return path.read_text().strip() if path.exists() else ""


def genome_names(root) -> list[str]:
    """Genomes present, in name order.

    Read from the files themselves rather than from a manifest — a manifest is
    one more thing to keep in step, and a stale one is worse than none.
    """
    out = []
    for path in layout.listing(Path(root) / layout.HMM_HITS, layout.TABLE_EXT):
        with layout.open_text(path) as fh:
            name = _read_name_line(fh.readline())
        out.append(name if name else layout.stem_of(path, layout.TABLE_EXT))
    return out


def read_proteins(root, genome: str) -> dict[str, str]:
    """Proteins for one genome, straight back from the stored FASTA."""
    path = layout.find(Path(root) / layout.PROTEINS, layout.safe(genome),
                       layout.FASTA_EXT)
    if path is None:
        raise FileNotFoundError(f"no stored proteins for {genome!r} under {root}")
    out: dict[str, str] = {}
    name, buf = None, []
    with layout.open_text(path) as fh:
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


def read_hits_for(root, genome: str) -> list[Hit]:
    """Raw hits for one genome."""
    path = layout.find(Path(root) / layout.HMM_HITS, layout.safe(genome),
                       layout.TABLE_EXT)
    if path is None:
        return []
    out = []
    with layout.open_text(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#") or line.startswith("protein\t"):
                continue
            prot, acc, score = line.rstrip("\n").split("\t")
            out.append(Hit(prot, acc, float(score)))
    return out


def read_hits(root) -> Iterator[tuple[str, list[Hit]]]:
    """Yield `(genome, hits)` for every genome, in name order."""
    for path in layout.listing(Path(root) / layout.HMM_HITS, layout.TABLE_EXT):
        with layout.open_text(path) as fh:
            named = _read_name_line(fh.readline())
        genome = named if named else layout.stem_of(path, layout.TABLE_EXT)
        yield genome, read_hits_for(root, genome)


def looks_like_archive(root) -> bool:
    p = Path(root)
    return (p / layout.PROTEINS).is_dir() and (p / layout.HMM_HITS).is_dir()


def size_report(root) -> dict[str, float]:
    root = Path(root)
    prot = sum(f.stat().st_size
               for f in layout.listing(root / layout.PROTEINS, layout.FASTA_EXT))
    hits = sum(f.stat().st_size
               for f in layout.listing(root / layout.HMM_HITS, layout.TABLE_EXT))
    return {"proteins_mb": prot / 1e6, "hits_mb": hits / 1e6,
            "total_mb": (prot + hits) / 1e6}
