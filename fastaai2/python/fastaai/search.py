"""HMM search and best-hit resolution.

The model set is loaded from an HMM file and *defines* the accession list —
accession IDs are positions in that list. There is no compiled-in Pfam set.

pyhmmer releases the GIL during search, so this is threaded rather than forked.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import sys
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


#: The model set shipped with the package — FastAAI 1's 122 SCPs, gzipped.
#:
#: A default, not a compiled-in set. It exists so an install works without
#: hunting for models and so databases built with defaults share one fingerprint,
#: but `--hmm` overrides it and everything downstream follows whichever file is
#: used. A v2 keyed to GTDB swaps this and gets a different fingerprint, so the
#: two generations refuse to be compared rather than mixing quietly.
BUNDLED_HMM = "Complete_SCG_DB.hmm.gz"


#: Marker sets shipped besides the default, selected by keyword instead of path.
#:
#: `--hmm` otherwise names a file, which makes a packaged set awkward to reach:
#: it lives inside site-packages at a path that varies by install. These names
#: resolve to the shipped files wherever they landed.
#:
#: Each maps to the files it is built from and a human label. `gtdb-all` names
#: both, because it is their *union*, assembled by `ModelSet` rather than
#: shipped as a third file. Nothing here is a default; the bundled 122 SCPs
#: remain what a bare `fastaai build` uses, and each of these sets fingerprints
#: differently, so databases built from different keywords refuse to compare.
GTDB_BAC120 = "gtdb_bac120.hmm.gz"
GTDB_AR53 = "gtdb_ar53.hmm.gz"

MODEL_SETS: dict[str, tuple[tuple[str, ...], str]] = {
    "gtdb-bact": ((GTDB_BAC120,), "GTDB bac120 (bacteria)"),
    "gtdb-arch": ((GTDB_AR53,), "GTDB ar53 (archaea)"),
    "gtdb-all": ((GTDB_BAC120, GTDB_AR53), "GTDB bac120 + ar53"),
}


def data_path(name: str) -> str:
    """Filesystem path to a file packaged in `fastaai/data`."""
    from importlib import resources

    return str(resources.files(__package__) / "data" / name)


def bundled_hmm_path() -> str:
    """Filesystem path to the packaged default model set."""
    return data_path(BUNDLED_HMM)


def model_set_key(spec: os.PathLike | str | None) -> str | None:
    """The `MODEL_SETS` key a `--hmm` value names, or None if it names a file.

    Single source of the normalisation rule, so callers that report which set
    was used agree with the one that loads it.

    `gtdb_bact`, `GTDB-BACT` and `gtdb-bact` are plainly the same request — case
    and separator are the two ways people habitually vary a name like this, and
    rejecting them teaches nothing. Anything holding a path separator is a path.
    """
    if spec is None:
        return None
    text = os.fspath(spec)
    if os.sep in text or (os.altsep and os.altsep in text):
        return None
    key = text.lower().replace("_", "-")
    return key if key in MODEL_SETS else None


def resolve_model_spec(spec: os.PathLike | str) -> list[str]:
    """A `--hmm` value to the HMM files it names.

    A keyword from `MODEL_SETS` wins over a file of the same name in the working
    directory. The alternative — letting a local file shadow the keyword — makes
    `--hmm gtdb-bact` mean different things in different directories, and the
    failure is silent: a stray file builds a database with a valid fingerprint
    that is simply not GTDB's. Shadowing is reported rather than obeyed, and a
    path that contains a separator (`./gtdb-bact`) is never read as a keyword.
    """
    text = os.fspath(spec)
    key = model_set_key(text)
    if key is not None:
        if os.path.exists(text):
            print(f"note: --hmm {text} is the packaged model set, not the file of "
                  f"that name here; use ./{text} for the file", file=sys.stderr)
        return [data_path(n) for n in MODEL_SETS[key][0]]

    # A near miss is a typo, not a filename. Saying "no such file" for `gtdb`
    # or `gtdb-bac` would send the user hunting for a download that does not
    # exist, when the set is already installed under a neighbouring name.
    if text.lower().replace("_", "-").startswith("gtdb") and not os.path.exists(text):
        raise SystemExit(
            f"--hmm {text}: no such model set or file. The packaged GTDB sets are\n"
            + "\n".join(f"  {k:<10} {len(MODEL_SETS[k][0])} file(s) — {MODEL_SETS[k][1]}"
                        for k in MODEL_SETS)
            + "\nThey fingerprint differently and cannot be compared with each other."
        )
    return [text]


def _open_hmm(path: str):
    """Open an HMM file, transparently decompressing a gzipped one.

    Detected by magic bytes rather than by suffix, so a user's own gzipped
    models work too. The bundled set ships compressed: 9.2 MB against 2.0 MB,
    and pyhmmer reads the stream without an intermediate file.
    """
    fh = open(path, "rb", buffering=HMM_READ_BUFFER)
    if fh.peek(2)[:2] == b"\x1f\x8b":
        fh.close()
        return gzip.open(path, "rb")
    return fh


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
    with _open_hmm(path) as inf:
        with pyhmmer.plan7.HMMFile(inf) as fh:
            return list(fh)


class ModelSet:
    """HMMs plus the accession ordering they induce."""

    def __init__(self, path: os.PathLike | str | None = None):
        #: None selects the bundled set, so callers need not know where it lives.
        #: A string may also be a keyword from `MODEL_SETS`, which can name more
        #: than one file.
        self.paths = [bundled_hmm_path()] if path is None else resolve_model_spec(path)
        #: The first source, kept because it is what a single-file set *is*.
        self.path = self.paths[0]
        self.alphabet = pyhmmer.easel.Alphabet.amino()

        loaded = [h for p in self.paths for h in _load_hmms(p)]
        if not loaded:
            raise ValueError(f"no HMMs found in {', '.join(self.paths)}")

        # Multi-file sets are a *union*, not a concatenation. bac120 and ar53
        # share 5 markers, and keeping both copies would break the invariant the
        # whole schema rests on — accession IDs are positions in this list, and
        # `acc_index` is keyed by accession, so a repeat silently makes one
        # position unaddressable and lets one protein occupy two slots.
        #
        # Only merges are deduplicated. A single file holding a repeated
        # accession is a defect in that file, and quietly dropping models would
        # hide it.
        self.hmms = []
        self.accessions: list[str] = []
        #: Accessions a multi-file set found in more than one of its sources.
        self.shared: list[str] = []
        seen: set[str] = set()
        for hmm in loaded:
            raw = hmm.accession or hmm.name
            acc = raw.decode() if isinstance(raw, bytes) else str(raw)
            if len(self.paths) > 1 and acc in seen:
                self.shared.append(acc)
                continue
            seen.add(acc)
            self.hmms.append(hmm)
            self.accessions.append(acc)
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

    Read through the buffer protocol, which pyhmmer's matrices support, rather
    than through numpy. The bytes are identical either way — verified across all
    122 bundled models — so this does not change any stored fingerprint; it just
    means the digest costs no dependency.
    """
    h = hashlib.sha256()
    for attr in ("match_emissions", "insert_emissions", "transition_probabilities"):
        mv = memoryview(getattr(hmm, attr))
        if mv.format != "f":
            raise TypeError(f"{attr}: expected float32 matrix, got format {mv.format!r}")
        h.update(str(tuple(mv.shape)).encode())
        h.update(mv.tobytes())
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
