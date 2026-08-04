"""HMM search and best-hit resolution.

The model set is loaded from an HMM file and *defines* the accession list —
accession IDs are positions in that list. There is no compiled-in Pfam set.

pyhmmer releases the GIL during search, so this is threaded rather than forked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import pyhmmer

FilterMode = Literal["v1", "v1_alt", "rbh"]

#: Resolution semantics for competing protein/model assignments.
#:
#: ``rbh``     strict reciprocal best hit: keep (P, A) only when A is P's best
#:             model AND P is A's best protein. This is the *stated intent* of
#:             FastAAI, and is markedly harsher than either shipped v1 path.
#: ``v1``      FastAAI 1 as actually executed (``pyhmmer_manager``): unique by
#:             protein, then unique by accession. Admits some non-reciprocal
#:             hits. Default, so results stay comparable to published numbers.
#: ``v1_alt``  the uncalled ``new_pyhmmer_manager`` ordering: unique by
#:             accession, then unique by protein. Neither a superset nor a
#:             subset of ``v1``.
#:
#: These give different SCP sets and therefore different AAI. The choice is
#: recorded in the database schema so a comparison cannot silently mix them.
DEFAULT_FILTER: FilterMode = "v1"


@dataclass(frozen=True)
class Hit:
    protein: str
    accession: str
    score: float


#: Read buffer for HMM parsing. Python's 8 KB default costs ~2.5x here purely in
#: small-read overhead; 1 MB brings streaming level with a full in-memory read.
HMM_READ_BUFFER = 1 << 20


def _load_hmms(path: str) -> list:
    """Parse an HMM file by handing pyhmmer an open file object.

    HMMER's own file reader is slow and pyhmmer binds it faithfully, so passing a
    *path* takes that slow route. Any Python file-like object bypasses it.
    Measured on the bundled 9.22 MB / 122-model set, median of 7:

        HMMFile(path)                       4121 ms
        open(rb) -> HMMFile(fileobj)         428 ms   (default 8 KB buffer)
        open(rb, 1 MB) -> HMMFile(fileobj)   173 ms   <- used here
        read() + HMMFile(BytesIO)            154 ms

    All routes yield identical models — same names, accessions and lengths.

    The buffered stream is chosen over the marginally faster full read because it
    holds constant memory regardless of model-database size, which matters for
    Pfam-scale files and removes any need for a fallback path.
    """
    with open(path, "rb", buffering=HMM_READ_BUFFER) as inf:
        with pyhmmer.plan7.HMMFile(inf) as fh:
            return list(fh)


class ModelSet:
    """HMMs plus the accession ordering they induce."""

    def __init__(self, path: os.PathLike | str):
        self.path = os.fspath(path)
        self.alphabet = pyhmmer.easel.Alphabet.amino()
        self.hmms = _load_hmms(self.path)
        if not self.hmms:
            raise ValueError(f"no HMMs found in {self.path}")

        self.accessions: list[str] = []
        for hmm in self.hmms:
            raw = hmm.accession or hmm.name
            self.accessions.append(raw.decode() if isinstance(raw, bytes) else str(raw))
        self.acc_index = {a: i for i, a in enumerate(self.accessions)}

        # `bit_cutoffs="trusted"` raises on models lacking TC lines, which
        # user-supplied models frequently are. Detect rather than assume.
        self.has_trusted = all(
            getattr(h.cutoffs, "trusted_available", lambda: False)() for h in self.hmms
        )

    def __len__(self) -> int:
        return len(self.hmms)

    @property
    def bit_cutoffs(self) -> str | None:
        return "trusted" if self.has_trusted else None


def _resolve(hits: list[Hit], mode: FilterMode) -> dict[str, str]:
    """Resolve competing assignments into protein -> accession."""
    # Stable sort so equal bit scores resolve deterministically. FastAAI 1 used
    # an unstable argsort then reversed it, making ties non-reproducible.
    ordered = sorted(hits, key=lambda h: (-h.score, h.protein, h.accession))

    if mode == "rbh":
        best_for_prot: dict[str, Hit] = {}
        best_for_acc: dict[str, Hit] = {}
        for h in ordered:
            if h.protein not in best_for_prot:
                best_for_prot[h.protein] = h
            if h.accession not in best_for_acc:
                best_for_acc[h.accession] = h
        return {
            h.protein: h.accession
            for h in ordered
            if best_for_prot[h.protein].accession == h.accession
            and best_for_acc[h.accession].protein == h.protein
        }

    if mode == "v1":
        first, second = "protein", "accession"
    elif mode == "v1_alt":
        first, second = "accession", "protein"
    else:
        raise ValueError(f"unknown filter mode {mode!r}")

    seen: set[str] = set()
    stage1 = []
    for h in ordered:
        key = getattr(h, first)
        if key not in seen:
            seen.add(key)
            stage1.append(h)

    seen.clear()
    out: dict[str, str] = {}
    for h in stage1:
        key = getattr(h, second)
        if key not in seen:
            seen.add(key)
            out[h.protein] = h.accession
    return out


def search_hits(
    proteins: dict[str, str],
    models: ModelSet,
    cpus: int = 1,
) -> list[Hit]:
    """Every *included* hit, unfiltered.

    Kept separate from resolution so the raw search output can be persisted:
    re-filtering under different semantics then costs nothing, where re-searching
    costs ~0.6 s/genome.
    """
    if not proteins:
        return []

    digital = [
        pyhmmer.easel.TextSequence(name=name.encode(), sequence=seq).digitize(
            models.alphabet
        )
        for name, seq in proteins.items()
    ]

    kwargs = {"cpus": cpus}
    if models.bit_cutoffs is not None:
        kwargs["bit_cutoffs"] = models.bit_cutoffs

    hits: list[Hit] = []
    for top in pyhmmer.hmmsearch(models.hmms, digital, **kwargs):
        query = getattr(top, "query", None)
        raw = getattr(query, "accession", None) or getattr(query, "name", None)
        if raw is None:
            raw = getattr(top, "query_name", b"")
        acc = raw.decode() if isinstance(raw, bytes) else str(raw)
        for hit in top:
            if not hit.included:
                continue
            name = hit.name.decode() if isinstance(hit.name, bytes) else str(hit.name)
            hits.append(Hit(name, acc, float(hit.score)))

    return hits


def resolve_hits(hits: list[Hit], mode: FilterMode = DEFAULT_FILTER) -> dict[str, str]:
    """Resolve raw hits into protein -> accession under *mode*."""
    return _resolve(hits, mode)


def best_hits(
    proteins: dict[str, str],
    models: ModelSet,
    mode: FilterMode = DEFAULT_FILTER,
    cpus: int = 1,
) -> dict[str, str]:
    """Search and resolve in one step; returns accession -> protein sequence."""
    hits = search_hits(proteins, models, cpus=cpus)
    assignment = _resolve(hits, mode)
    return {acc: proteins[prot] for prot, acc in assignment.items()}
