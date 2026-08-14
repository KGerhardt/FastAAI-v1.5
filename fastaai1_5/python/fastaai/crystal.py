"""Crystals — the resolved-SCP rank of preprocessing output.

FastAAI 1 called these crystals and this is the same idea, carried over. There
are three ranks of intermediate output, each reusable across a different kind of
change:

    proteins/<genome>.fasta          every gene call    reusable across model sets
    hmm_hits/<genome>.tsv            every raw hit      reusable across filter modes
    crystals/<genome>.crystal.fasta  the SCPs that won  reusable across nothing — but
                                                        it is what a database is built
                                                        from, and it is tiny

The first two are the archive (see `archive.py`). This is the third. Measured on
2,943 Firmicutes: 9.6 KB per genome against 543 KB for the full protein set, so
a collection is 29 MB where its archive is 1.7 GB and the *database built from
it* is 117 MB. A crystal set is the smallest of the three artifacts and the only
one that is both shippable and directly buildable, which makes it the right unit
to distribute, subset and organise by hand.

**Why this rank earns its place.** Building a database is the only thing you can
do with crystals, and it is fast because everything expensive already happened.
That makes rebuilding cheap enough to be the answer to database growth: rather
than appending genomes to a sealed database — which fragments it into
one-genome partitions and costs ~90x search throughput — you keep crystals and
rebuild when you are ready. There is deliberately no incremental append.

**This is the ingestion path, not a side output.** Every preprocessing worker
writes its own crystal and the database is always built by reading them back.
Crystals are always kept, like every other rank — they land in
`<root>/crystals/` and nothing is thrown away. One path in, so there is nothing
to keep in step. It also decouples peak memory from collection size twice over:
a worker drops a genome's sequences the moment its crystal is on disk, and the
build streams them back one file at a time. Ingesting proteins, or stored
proteins plus raw hits, is still supported — those are earlier ranks, and they
feed this one.

**Format.** One file per genome, FASTA, one record per SCP (gzipped only if
`--gzip` was given)::

    >PF00380.26 genome=GCF_000007085.1 protein=NC_004116.1_1523 models=45d1… filter=v1 table=11
    MKVLAATT…

Two choices worth stating, because both are load-bearing:

*Accession names, not positions.* Accession IDs are positions in a model list,
but a crystal stores the accession's name. Position is assigned at build time
from the model set, so a crystal does not depend on model ordering and any
subset of crystals still builds a database comparable with any other subset
from the same models.

*Provenance repeated on every record, not once per file.* The redundancy is
what makes `cat a.crystal.fasta b.crystal.fasta` and arbitrary splitting safe —
the unit of meaning is the record, so no operation on whole files can strand
metadata. It costs a few hundred bytes a genome, and nothing at all under
`--gzip`.

`models=` is the model-set fingerprint. A build refuses crystals whose
fingerprint disagrees with the model set it was given, which is the guard that
stops a database being assembled from two different marker sets and reporting
numerically valid, biologically meaningless AAI.

**Genome order is by name, deliberately.** A crystal build sorts, so the same
crystals always produce the same genome ordinals no matter what order the files
were enumerated in — rebuilds are reproducible and diffable. An archive build
instead preserves the archive's own order, so the two routes over one collection
emit the same rows in a different order. Verified on 2,943 Firmicutes: all
8,661,250 rows identical as sets, differing only in sequence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

from . import layout

SUFFIX = layout.CRYSTAL_EXT

#: Wrap sequences at this width. FASTA convention, and it keeps a crystal
#: readable in a pager rather than as one enormous line.
WRAP = 60


def _safe(name: str) -> str:
    """A genome name as a filename. Mirrors `archive._safe`."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _fmt(seq: str) -> str:
    return "\n".join(seq[i:i + WRAP] for i in range(0, len(seq), WRAP)) or ""


def _enc(value: str) -> str:
    """Percent-encode a header value.

    Fields are whitespace-delimited `key=value`, so a genome named
    `my genome v2` would otherwise parse back as `my` — silently, and the
    truncated name becomes the genome's identity everywhere downstream. Names
    come from filenames, which routinely carry spaces. Ordinary accessions and
    NCBI-style names contain only characters `quote` leaves alone, so this is
    invisible in practice and decisive when it is not.
    """
    from urllib.parse import quote

    return quote(str(value), safe="._-~")


def _dec(value: str) -> str:
    from urllib.parse import unquote

    return unquote(value)


def header(accession: str, genome: str, protein: str | None, fingerprint: str,
           filter_mode: str, table: int | None) -> str:
    fields = [f"genome={_enc(genome)}"]
    if protein:
        fields.append(f"protein={_enc(protein)}")
    fields.append(f"models={_enc(fingerprint)}")
    fields.append(f"filter={_enc(filter_mode)}")
    if table is not None:
        fields.append(f"table={table}")
    return f">{_enc(accession)} " + " ".join(fields)


def render(genome: str, scps: dict[str, str], fingerprint: str, filter_mode: str,
           table: int | None = None,
           origins: dict[str, str] | None = None) -> str:
    """One genome's crystal, as text.

    Accessions are written in sorted order so a crystal is reproducible: the
    same genome and models give a byte-identical file regardless of the order
    HMMER happened to report hits in.
    """
    origins = origins or {}
    out = []
    for acc in sorted(scps):
        out.append(header(acc, genome, origins.get(acc), fingerprint,
                          filter_mode, table))
        out.append(_fmt(scps[acc]))
    return "\n".join(out) + ("\n" if out else "")


def write(root, genome: str, scps: dict[str, str], fingerprint: str,
          filter_mode: str, table: int | None = None,
          origins: dict[str, str] | None = None,
          compress: bool = False) -> Path | None:
    """Write one genome's crystal. Returns None for a genome with no SCPs.

    A genome that recovered nothing has no crystal rather than an empty one —
    an empty file is indistinguishable from a truncated write, and it would
    later look like a genome that legitimately built to nothing.
    """
    if not scps:
        return None
    text = render(genome, scps, fingerprint, filter_mode, table, origins)
    return layout.write_text(
        Path(root) / f"{layout.safe(genome)}{SUFFIX}", text, compress)


def _open(path):
    return layout.open_text(path)


def parse_header(line: str) -> tuple[str, dict[str, str]]:
    """`>ACC key=value ...` to (accession, fields)."""
    body = line[1:].strip()
    if not body:
        raise ValueError("empty FASTA header in crystal")
    parts = body.split()
    acc, fields = _dec(parts[0]), {}
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k] = _dec(v)
    return acc, fields


def read_file(path) -> Iterator[tuple[str, str, str, dict[str, str]]]:
    """Yield (genome, accession, sequence, fields) from one crystal file.

    Grouping is by the `genome=` field rather than by file, so a concatenation
    of many crystals reads back as many genomes. That is the property that makes
    them freely organisable.
    """
    acc = genome = None
    fields: dict[str, str] = {}
    chunks: list[str] = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    yield genome, acc, "".join(chunks), fields
                acc, fields = parse_header(line)
                genome = fields.get("genome")
                if not genome:
                    raise ValueError(
                        f"{path}: record {acc!r} has no genome= field; this is "
                        "not a FastAAI crystal"
                    )
                chunks = []
            elif line.strip():
                chunks.append(line.strip())
    if acc is not None:
        yield genome, acc, "".join(chunks), fields


def crystal_paths(source) -> list[Path]:
    """Crystal files at *source*, which may be a directory or a single file."""
    p = Path(source)
    if p.is_file():
        return [p]
    # A run's crystals live in `<root>/crystals`, so accept the root as well as
    # the directory itself — `fastaai build FastAAI/` should mean the obvious
    # thing.
    if (p / layout.CRYSTALS).is_dir() and layout.listing(p / layout.CRYSTALS, SUFFIX):
        p = p / layout.CRYSTALS
    return layout.listing(p, SUFFIX)


def looks_like_crystals(source) -> bool:
    p = Path(source)
    if p.is_file():
        name = os.fspath(p)
        return name.endswith(SUFFIX) or name.endswith(SUFFIX + ".gz")
    return p.is_dir() and bool(crystal_paths(p))


class Provenance:
    """The model set and filter every crystal in a run must agree on.

    Accumulated as files stream past rather than gathered up front, so the check
    costs nothing and still fires on the first disagreement.
    """

    def __init__(self):
        self.models: set[str] = set()
        self.filter: set[str] = set()

    def observe(self, fields: dict[str, str]) -> None:
        if "models" in fields:
            self.models.add(fields["models"])
            if len(self.models) > 1:
                raise ValueError(
                    "crystals disagree on the model set: "
                    + ", ".join(sorted(self.models))
                    + ". They cannot build one database."
                )
        if "filter" in fields:
            self.filter.add(fields["filter"])
            if len(self.filter) > 1:
                raise ValueError(
                    "crystals disagree on the best-hit filter: "
                    + ", ".join(sorted(self.filter))
                    + ". Different filters give different SCP sets and "
                    "different AAI."
                )

    def as_dict(self) -> dict[str, str]:
        return {"models": next(iter(self.models), ""),
                "filter": next(iter(self.filter), "")}


def iter_genomes(source) -> Iterator[tuple[str, dict[str, str], Provenance]]:
    """Stream crystals one genome at a time.

    Memory is bounded by the largest single *file*, not by the collection, which
    is what lets a build scale past what would fit in RAM — a 500k-genome
    collection would otherwise materialise every SCP sequence at once.

    Grouping is still by `genome=`, so a concatenated file yields each of its
    genomes. The bounded-memory version of that has one consequence worth
    stating: a genome whose records are spread across *different* files cannot
    be reassembled without holding everything, so it is rejected rather than
    silently added twice.
    """
    prov = Provenance()
    seen_elsewhere: set[str] = set()

    paths = crystal_paths(source)
    if not paths:
        raise ValueError(f"no crystals ({SUFFIX}) found at {source}")

    for path in paths:
        here: dict[str, dict[str, str]] = {}
        for genome, acc, seq, fields in read_file(path):
            prov.observe(fields)
            if genome in seen_elsewhere:
                raise ValueError(
                    f"{path}: genome {genome!r} also appears in an earlier "
                    "file. Crystals for one genome must be in one file."
                )
            bucket = here.setdefault(genome, {})
            if acc in bucket and bucket[acc] != seq:
                raise ValueError(
                    f"{path}: genome {genome!r} has two different sequences for "
                    f"accession {acc}"
                )
            bucket[acc] = seq
        for genome in sorted(here):
            seen_elsewhere.add(genome)
            yield genome, here[genome], prov


def read(source) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Read crystals into {genome: {accession: sequence}} plus their provenance.

    The materialising form, for callers small enough not to care. A build uses
    `iter_genomes` instead.
    """
    genomes: dict[str, dict[str, str]] = {}
    prov = Provenance()
    for genome, scps, prov in iter_genomes(source):
        genomes[genome] = scps
    return genomes, prov.as_dict()


def write_many(root, records: Iterable, fingerprint: str, filter_mode: str) -> int:
    """Write crystals for an iterable of `pipeline.GenomeRecord`."""
    n = 0
    for rec in records:
        if getattr(rec, "error", None) or not rec.scps:
            continue
        if write(root, rec.name, rec.scps, fingerprint, filter_mode,
                 rec.translation_table, getattr(rec, "scp_proteins", None)):
            n += 1
    return n
