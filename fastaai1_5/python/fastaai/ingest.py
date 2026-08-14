"""FASTA ingestion.

One code path for plain and gzipped input. ``pyfastx.Fastx`` is the lightweight
sequential reader; ``pyfastx.Fasta`` carries index machinery we do not want —
measured 2x slower here and it can leave ``.fxi`` sidecars beside read-only data.

pyfastx sits at ~0.86x a hand-rolled gzip parser for a single pass, which is
irrelevant: ingestion runs ~0.03 s/genome against ~4.8 s/genome for prediction
plus HMM search. Robustness and a single code path decide it.

Note ``pyfastx`` requires a *path*. Passing bytes segfaults the interpreter with
no traceback, so ``read_fasta`` refuses anything that is not a path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pyfastx

#: Extensions treated as FASTA, with or without a compression suffix.
FASTA_SUFFIXES = (".fna", ".fa", ".fasta", ".faa", ".fas", ".fa.gz")

#: Extensions conventionally meaning "amino acid", i.e. already-predicted
#: proteins. Used only as a default when the caller does not say which it has.
PROTEIN_SUFFIXES = (".faa", ".fasta.aa", ".aa")
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zst")


def looks_like_fasta(path: os.PathLike | str) -> bool:
    """True if *path* has a FASTA extension, ignoring any compression suffix."""
    name = Path(path).name.lower()
    for comp in COMPRESSION_SUFFIXES:
        if name.endswith(comp):
            name = name[: -len(comp)]
            break
    return name.endswith(FASTA_SUFFIXES)


def read_fasta(path: os.PathLike | str) -> Iterator[tuple[str, str]]:
    """Yield ``(name, sequence)`` for each record, decompressing transparently.

    Raises TypeError for non-path input rather than letting pyfastx crash.
    """
    if isinstance(path, (bytes, bytearray, memoryview)):
        raise TypeError(
            "read_fasta requires a filesystem path; pyfastx segfaults on raw bytes"
        )
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    yield from pyfastx.Fastx(path)


def read_sequence(path: os.PathLike | str) -> str:
    """Concatenate every record into one string. Used for training gene models."""
    return "".join(seq for _, seq in read_fasta(path))


def genome_name(path: os.PathLike | str) -> str:
    """Strip FASTA and compression suffixes to get a display name."""
    name = Path(path).name
    for comp in COMPRESSION_SUFFIXES:
        if name.lower().endswith(comp):
            name = name[: -len(comp)]
            break
    stem = Path(name).stem
    return stem or name


def looks_like_protein(path: os.PathLike | str) -> bool:
    """True if the extension conventionally means amino acids.

    A guess, not a guarantee — `.fasta` is used for both. Callers that know
    should say so explicitly rather than rely on this.
    """
    name = Path(path).name.lower()
    for comp in COMPRESSION_SUFFIXES:
        if name.endswith(comp):
            name = name[: -len(comp)]
            break
    return name.endswith(PROTEIN_SUFFIXES)


def read_proteins_fasta(path: os.PathLike | str) -> dict[str, str]:
    """Read a protein FASTA into `{gene_id: sequence}`.

    Gene ids are the first whitespace-delimited token of the defline, matching
    how predicted proteins are conventionally named. Duplicate ids are an error:
    silently keeping the last would drop a marker without telling anyone.
    """
    out: dict[str, str] = {}
    for name, seq in read_fasta(path):
        gene = name.split()[0] if name else name
        if gene in out:
            raise ValueError(f"{path}: duplicate sequence id {gene!r}")
        out[gene] = seq
    return out


def find_genomes(root: os.PathLike | str, recursive: bool = True) -> list[Path]:
    """All FASTA files under *root*, sorted for reproducible genome ordering.

    Ordering matters: it becomes the row/column order of the output matrix.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    walker = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in walker if p.is_file() and looks_like_fasta(p))
