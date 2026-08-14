"""Assemble GTDB's bac120 and ar53 marker sets from their primary sources.

GTDB does not distribute these HMMs on their own — they are inside the GTDB-Tk
reference package, which is ~100 GB. But the marker *lists* are published, and
every marker is a Pfam or TIGRFAM family available separately, so the sets can be
rebuilt from about 15 MB of traffic.

    bac120   6 Pfam + 114 TIGRFAM
    ar53    12 Pfam +  41 TIGRFAM

Sources:
    marker lists  data.gtdb.ecogenomic.org/releases/latest/auxillary_files/
    Pfam          EBI InterPro API, one family at a time
    TIGRFAM       NCBI, which absorbed TIGRFAMs into the PGAP library
                  (the JCVI mirror is gone)

**This is not a faithful reproduction of GTDB's own files.** The marker list
pins Pfam versions — R232 asks for PF00380.20 — and InterPro serves only the
current one, .26 at time of writing. Same family, later model. For benchmarking
that is fine; for reproducing GTDB's published trees it is not, and the
fingerprint on any database built from this will differ from one built on
GTDB's actual files. The versions actually fetched are recorded in
`<set>_versions.tsv` beside the output.

    python fetch_gtdb_markers.py [outdir]
"""

from __future__ import annotations

import gzip
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GTDB = "https://data.gtdb.ecogenomic.org/releases/latest/auxillary_files"
PGAP = "https://ftp.ncbi.nlm.nih.gov/hmm/15.0"
INTERPRO = "https://www.ebi.ac.uk/interpro/api/entry/pfam"

SETS = {"bac120": "bac120_msa_marker_info.tsv", "ar53": "ar53_msa_marker_info.tsv"}


def get(url: str, tries: int = 3) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"    retry {attempt + 1}: {url} ({e})", file=sys.stderr)
    raise AssertionError("unreachable")


def markers(out: Path, which: str) -> list[str]:
    """Marker IDs for one set, as GTDB writes them (PFAM_PF..., TIGR_TIGR...)."""
    path = out / SETS[which]
    if not path.exists():
        path.write_bytes(get(f"{GTDB}/{SETS[which]}"))
    rows = path.read_text().rstrip("\n").split("\n")[1:]
    return [r.split("\t")[0] for r in rows]


def pgap_index(out: Path) -> dict[str, str]:
    """TIGR accession -> versioned accession, e.g. TIGR00006 -> TIGR00006.1.

    NCBI names the model files by versioned accession, so the bare TIGR number
    404s. The mapping is only in this 13 MB table.
    """
    path = out / "hmm_PGAP.tsv"
    if not path.exists():
        print("  fetching PGAP accession table (13 MB)...", file=sys.stderr)
        path.write_bytes(get(f"{PGAP}/hmm_PGAP.tsv"))
    index = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            f = line.split("\t")
            if len(f) > 1 and f[1].startswith("TIGR"):
                index[f[1]] = f[0]
    return index


def fetch_one(marker: str, tigr: dict[str, str]) -> tuple[bytes, str]:
    """One model's HMM text, plus the version actually obtained."""
    if marker.startswith("PFAM_"):
        acc = marker[len("PFAM_"):].split(".")[0]
        raw = gzip.decompress(get(f"{INTERPRO}/{acc}/?annotation=hmm"))
    elif marker.startswith("TIGR_"):
        acc = marker[len("TIGR_"):]
        versioned = tigr.get(acc)
        if versioned is None:
            raise KeyError(f"{acc} not in the PGAP table")
        raw = get(f"{PGAP}/hmm_PGAP.HMM/{versioned}.HMM")
    else:
        raise ValueError(f"unrecognised marker id {marker!r}")

    got = ""
    for line in raw.decode("utf-8", "replace").split("\n"):
        if line.startswith("ACC "):
            got = line.split()[1]
            break
    if not raw.rstrip().endswith(b"//"):
        raw = raw.rstrip() + b"\n//\n"
    return raw, got


def build(which: str, out: Path, tigr: dict[str, str]) -> Path:
    ids = markers(out, which)
    print(f"{which}: {len(ids)} markers", file=sys.stderr)

    dest = out / f"{which}.hmm"
    versions = []
    with open(dest, "wb") as fh:
        for i, marker in enumerate(ids, 1):
            raw, got = fetch_one(marker, tigr)
            fh.write(raw if raw.endswith(b"\n") else raw + b"\n")
            versions.append((marker, got))
            if i % 20 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)}", file=sys.stderr)

    # GTDB pins a version for Pfam (PFAM_PF00380.20) but not for TIGRFAM
    # (TIGR_TIGR00006). The `.1` on a fetched TIGR accession is NCBI's own
    # suffix, so counting it as drift would report every marker as changed.
    def drift(marker: str, got: str) -> str:
        want = marker.split("_", 1)[1]
        if "." not in want:
            return "unpinned"
        return "yes" if got and got != want else "no"

    with open(out / f"{which}_versions.tsv", "w") as fh:
        fh.write("gtdb_marker_id\tfetched_accession\tdrifted\n")
        for marker, got in versions:
            fh.write(f"{marker}\t{got}\t{drift(marker, got)}\n")

    drifted = sum(1 for m, g in versions if drift(m, g) == "yes")
    pinned = sum(1 for m, _ in versions if "." in m.split("_", 1)[1])
    print(f"  wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB); "
          f"{drifted}/{pinned} version-pinned markers differ from GTDB "
          f"({len(ids) - pinned} unpinned)", file=sys.stderr)
    return dest


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "gtdb_markers")
    out.mkdir(parents=True, exist_ok=True)
    tigr = pgap_index(out)
    print(f"  PGAP table: {len(tigr):,} TIGR accessions\n", file=sys.stderr)
    for which in SETS:
        build(which, out, tigr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
