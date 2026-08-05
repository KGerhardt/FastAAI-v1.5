"""HMM search and best-hit resolution.

The model set is loaded from an HMM file and *defines* the accession list —
accession IDs are positions in that list. There is no compiled-in Pfam set.

pyhmmer releases the GIL during search, so this is threaded rather than forked.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Literal

import pyhmmer

FilterMode = Literal["v1", "v1_alt", "rbh"]

#: Which HMMER score resolves competing hits.
#:
#: ``sequence``   full-sequence bit score, unrounded.
#: ``domain_v1``  best-domain bit score **rounded to 1 decimal** — what FastAAI 1
#:                actually used (`fastaai.py:359`, `390`). The rounding is
#:                incidental: v1 computed it to format a HMMER-style output table
#:                and then reused the display value as the sort key. Rounding
#:                manufactures ties, which then resolve by sort order rather than
#:                by score, so it is not a neutral choice.
#: ``domain``     best-domain bit score, unrounded.
ScoreKind = Literal["sequence", "domain_v1", "domain"]
DEFAULT_SCORE: ScoreKind = "domain_v1"

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
    #: Score used for best-hit resolution. Which score this is matters — see
    #: `ScoreKind` and `search_hits`.
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
    def fingerprint(self) -> str:
        """Identity of this model set, independent of implementation and file.

        Accession names and their order do not establish that two databases were
        built from the same models: a Pfam version bump or a locally edited HMM
        keeps every name and position while changing which proteins hit, which
        changes the k-mer sets and so the AAI — silently, because nothing about
        the output looks wrong.

        Hashed per model rather than over the file, so it survives reformatting,
        HMMER version differences and concatenation order of the source files:

        - the model's own parameters — match and insert emissions and transition
          probabilities. This is the model, so it is always computable and
          always meaningful.
        - `CKSUM`, HMMER's checksum of the training alignment, when the file
          carries one. Free identity from HMMER itself, but optional in the
          format, which is why it cannot be the only ingredient: a model without
          one would otherwise fingerprint on its name alone.
        - model length and both names, so models that somehow agree numerically
          still separate.

        Order is included by construction: accession IDs are positions in this
        list, so two sets holding the same models in a different order are not
        interchangeable and must not share a fingerprint.
        """
        h = hashlib.sha256()
        for acc, hmm in zip(self.accessions, self.hmms):
            cksum = hmm.checksum
            h.update("\t".join([
                acc,
                _text(hmm.name) or "",
                str(hmm.M),
                "-" if cksum is None else str(cksum),
                _model_digest(hmm),
            ]).encode())
            h.update(b"\n")
        return h.hexdigest()

    @property
    def bit_cutoffs(self) -> str | None:
        return "trusted" if self.has_trusted else None


def _model_digest(hmm) -> str:
    """Digest of a model's parameters — what actually decides which proteins hit.

    `CKSUM` is optional in the HMM format, so a fingerprint resting on it alone
    degrades to the model's name for any file that omits one. The parameters are
    always present, so this is always computable.

    Emissions and transitions are stored in the file as fixed-precision text and
    parsed to IEEE-754 floats, so the bytes are reproducible for anyone reading
    the same models.
    """
    import numpy as np

    h = hashlib.sha256()
    for attr in ("match_emissions", "insert_emissions", "transition_probabilities"):
        arr = np.ascontiguousarray(np.asarray(getattr(hmm, attr), dtype=np.float32))
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _text(v) -> str | None:
    if v is None:
        return None
    return v.decode() if isinstance(v, bytes) else str(v)


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
    score: ScoreKind = DEFAULT_SCORE,
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
        for hit in top:
            # v1 takes the accession off the best domain's alignment rather than
            # the TopHits query object (fastaai.py:353).
            raw = _text(hit.best_domain.alignment.hmm_accession)
            if raw is None:
                q = getattr(top, "query", None)
                raw = _text(getattr(q, "accession", None)) or _text(getattr(q, "name", None))
            if raw is None:
                continue
            if score == "sequence":
                sc = float(hit.score)
            elif score == "domain":
                sc = float(hit.best_domain.alignment.domain.score)
            else:  # domain_v1
                sc = round(float(hit.best_domain.alignment.domain.score), 1)
            hits.append(Hit(_text(hit.name), str(raw), sc))

    return hits


def resolve_hits(hits: list[Hit], mode: FilterMode = DEFAULT_FILTER) -> dict[str, str]:
    """Resolve raw hits into protein -> accession under *mode*."""
    return _resolve(hits, mode)


def best_hits(
    proteins: dict[str, str],
    models: ModelSet,
    mode: FilterMode = DEFAULT_FILTER,
    cpus: int = 1,
    score: ScoreKind = DEFAULT_SCORE,
) -> dict[str, str]:
    """Search and resolve in one step; returns accession -> protein sequence."""
    hits = search_hits(proteins, models, cpus=cpus, score=score)
    assignment = _resolve(hits, mode)
    return {acc: proteins[prot] for prot, acc in assignment.items()}
