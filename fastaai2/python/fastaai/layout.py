"""Where a run's output goes, and whether it is compressed.

Everything a run produces lands under one root — `FastAAI/` in the working
directory unless `--dir` says otherwise — as flat directories of one file per
genome::

    <root>/
      proteins/<genome>.fasta          every gene call
      hmm_hits/<genome>.tsv            every raw HMM hit
      crystals/<genome>.crystal.fasta  the SCPs that won
      database/<name>/                 the built database
      results/                         AAI output, from a search

**Intermediates are written by default, not on request.** Preprocessing is ~98%
of a cold run, so discarding it is the expensive choice; the flags that used to
opt in to keeping proteins and crystals are gone.

Three rules, and the reasons they are rules:

*One file per genome, no bundling.* No tar, no single concatenated table
covering the collection, no nesting beyond the one level above. A directory of
per-genome files can be listed, subsetted, copied and inspected with ordinary
tools, and one genome can be deleted or re-run without rewriting anything else.

*The working directory is the root.* Scratch space is not assumed to exist and
temporary directories are not used anywhere in this codebase. An HPC job is
given a directory and is expected to work inside it; writing elsewhere is how a
job fails on someone else's cluster.

*Compression is a flag, off by default.* `--gzip` gzips each file as it is
written, in memory, one file at a time. Plain text is the default because it is
inspectable; nothing downstream cares, since every reader here accepts either
form.

Output naming is ours and is standardised — FASTA is `.fasta`, tables are
`.tsv`. Input naming is the user's: `ingest.py` accepts `.faa`, `.fna`, `.fa`
and the rest, and nothing here requires a convention of anyone's own files.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Iterator

DEFAULT_ROOT = "FastAAI"

PROTEINS = "proteins"
HMM_HITS = "hmm_hits"
CRYSTALS = "crystals"
DATABASE = "database"
RESULTS = "results"

#: Our output extensions. FASTA is `.fasta` throughout — `.faa` is a name users
#: give their own inputs, not one we produce.
FASTA_EXT = ".fasta"
TABLE_EXT = ".tsv"
CRYSTAL_EXT = ".crystal.fasta"


def safe(name: str) -> str:
    """A genome name as a filename."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def open_text(path):
    """Open one of our files for reading, gzipped or not.

    Detected by magic bytes rather than by suffix, so a file someone gzipped or
    gunzipped by hand still reads.
    """
    fh = open(path, "rb")
    magic = fh.peek(2)[:2] if hasattr(fh, "peek") else b""
    fh.close()
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "r")


def write_text(path, text: str, compress: bool = False) -> Path:
    """Write one file, gzipping in memory when asked. Returns the path written.

    `.gz` is appended when compressing, so the name always says what the file
    is.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        path = path.with_name(path.name + ".gz")
        with gzip.open(path, "wt", compresslevel=9) as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def find(directory, stem: str, ext: str) -> Path | None:
    """Locate `<stem><ext>` or `<stem><ext>.gz`, whichever exists."""
    plain = Path(directory) / f"{stem}{ext}"
    if plain.exists():
        return plain
    gz = plain.with_name(plain.name + ".gz")
    return gz if gz.exists() else None


def listing(directory, ext: str) -> list[Path]:
    """Every `*<ext>` or `*<ext>.gz` in *directory*, sorted."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(list(d.glob(f"*{ext}")) + list(d.glob(f"*{ext}.gz")))


def stem_of(path, ext: str) -> str:
    """The genome part of one of our filenames."""
    name = Path(path).name
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    if name.endswith(ext):
        name = name[: -len(ext)]
    return name


class Layout:
    """The output root for one run, and the compression choice."""

    def __init__(self, root=DEFAULT_ROOT, compress: bool = False):
        self.root = Path(root)
        self.compress = compress

    def sub(self, name: str, create: bool = False) -> Path:
        p = self.root / name
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def proteins(self) -> Path:
        return self.sub(PROTEINS)

    @property
    def hmm_hits(self) -> Path:
        return self.sub(HMM_HITS)

    @property
    def crystals(self) -> Path:
        return self.sub(CRYSTALS)

    @property
    def database(self) -> Path:
        return self.sub(DATABASE)

    @property
    def results(self) -> Path:
        return self.sub(RESULTS)

    def database_path(self, name: str) -> Path:
        """Where a database called *name* lives.

        A bare name goes under `<root>/database/`, which is what makes `-d`
        mean "call it this" rather than "put it exactly here". A name carrying a
        path separator is taken literally, so an explicit location always wins.
        """
        if os.sep in str(name) or (os.altsep and os.altsep in str(name)):
            return Path(name)
        return self.database / name

    def __repr__(self) -> str:
        return f"Layout({str(self.root)!r}, compress={self.compress})"
