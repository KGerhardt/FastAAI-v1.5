"""Turning block output back into FastAAI 1's shapes.

v1 wrote results three ways: one monolithic TSV, one TSV per query genome
holding that genome against everything, and one monolithic matrix of AAI
values. v1.5 writes a grid of `(query partition x target partition)` blocks
instead, because the monolith is what stops being representable first — 200k
genomes square is 40 billion cells.

The block set is a superset of all three, so this reshapes it back:

    concatenate the blocks        -> the monolithic TSV (every block carries the
                                     same header, so `cat` almost does it)
    per_genome()                  -> one TSV per query genome
    to_matrix()                   -> the AAI-only matrix

Both functions stream. A query genome's rows sit contiguously within each block,
and the blocks of one query partition hold successive stretches of its targets,
so a row can be assembled by reading the blocks of that partition in step. Peak
memory is one row, not one matrix.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from . import _core, layout

#: `block_q00000_t00001.tsv` -> (0, 1). The names are the ordering: query
#: partition major, target partition minor, both zero-padded.
BLOCK = re.compile(r"block_q(\d+)_t(\d+)\.(tsv|matrix)$")


def block_files(source) -> list[tuple[int, int, Path]]:
    """`(query partition, target partition, path)` for a results directory.

    A single file is accepted too, so a 1x1 search — which lands in one file
    rather than a directory — reshapes the same way.
    """
    p = Path(source)
    if p.is_file():
        m = BLOCK.search(p.name)
        return [(int(m.group(1)), int(m.group(2)), p) if m else (0, 0, p)]
    out = []
    for f in sorted(p.iterdir()):
        m = BLOCK.search(f.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), f))
    if not out:
        raise ValueError(f"no block files (block_qNNNNN_tNNNNN.tsv) under {p}")
    return out


def _rows(path) -> Iterator[list[str]]:
    with layout.open_text(path) as fh:
        header = fh.readline()
        if not header.startswith("query\t"):
            raise ValueError(f"{path}: not a FastAAI result TSV")
        for line in fh:
            if line.strip():
                yield line.rstrip("\n").split("\t")


def _grouped(path) -> Iterator[tuple[str, list[list[str]]]]:
    """Rows of one block, grouped by query genome, in file order."""
    current, batch = None, []
    for f in _rows(path):
        if f[0] != current:
            if current is not None:
                yield current, batch
            current, batch = f[0], []
        batch.append(f)
    if current is not None:
        yield current, batch


def header_of(path) -> str:
    with layout.open_text(path) as fh:
        return fh.readline().rstrip("\n")


def per_genome(source, out_dir, *, compress: bool = False) -> list[Path]:
    """One TSV per query genome, that genome against everything — v1's shape.

    Each file carries the block header, so it parses exactly as a v1 per-genome
    output did.

    Written a genome at a time. Holding a file handle per genome would mean
    16,384 of them for one partition, which is past most open-file limits, so
    each genome's rows are gathered across its partition's blocks and written
    once.
    """
    blocks = block_files(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = header_of(blocks[0][2])

    by_q: dict[int, list[Path]] = {}
    for qi, ti, path in blocks:
        by_q.setdefault(qi, []).append(path)

    written: list[Path] = []
    for qi in sorted(by_q):
        # Blocks of one query partition, in target order. Step them together so
        # each genome's targets come out in the order a single search produced.
        streams = [_grouped(p) for p in by_q[qi]]
        for groups in zip(*streams):
            names = {g[0] for g in groups}
            if len(names) != 1:
                raise ValueError(
                    f"blocks of query partition {qi} disagree on row order: {names}"
                )
            genome = groups[0][0]
            body = "".join("\t".join(f) + "\n" for _, rows in groups for f in rows)
            written.append(layout.write_text(
                out / f"{layout.safe(genome)}.tsv", head + "\n" + body, compress))
    return written


def to_matrix(source, out_path, *, compress: bool = False) -> Path:
    """The AAI-only matrix, from TSV blocks — v1's third output.

    The TSV's `AAI_estimate` is a label as often as a number: `N/A` where no
    marker is shared, `<30%` and `>90%` where the regression cannot resolve. A
    matrix cell holds a number, so those carry v1's sentinel values instead. The
    mapping is the engine's own (`_core.matrix_cell_from_label`) rather than a
    copy of it.

    Streams one query row at a time, so the matrix is never resident.
    """
    blocks = block_files(source)
    cols = [c for c in header_of(blocks[0][2]).split("\t")]
    try:
        aai_col = cols.index("AAI_estimate")
    except ValueError:
        raise ValueError(
            f"{blocks[0][2]}: no AAI_estimate column — was this written with "
            "--emit jaccard?"
        ) from None

    by_q: dict[int, list[Path]] = {}
    for qi, ti, path in blocks:
        by_q.setdefault(qi, []).append(path)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dest = out.with_name(out.name + ".gz") if compress else out

    import gzip

    opener = (lambda: gzip.open(dest, "wt", compresslevel=9)) if compress \
        else (lambda: open(dest, "w"))
    with opener() as fh:
        wrote_header = False
        for qi in sorted(by_q):
            streams = [_grouped(p) for p in by_q[qi]]
            for groups in zip(*streams):
                genome = groups[0][0]
                cells, targets = [], []
                for _, rows in groups:
                    for f in rows:
                        targets.append(f[1])
                        cells.append(_core.matrix_cell_from_label(f[aai_col]))
                if not wrote_header:
                    fh.write("query_genome\t" + "\t".join(targets) + "\n")
                    wrote_header = True
                fh.write(genome + "\t" + "\t".join(cells) + "\n")
    return dest
