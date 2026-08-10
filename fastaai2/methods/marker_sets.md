# Swapping the SCP set

400 Firmicutes genomes, one archive, three marker sets. Gene prediction is not
repeated — only HMM search, which is the stage a marker swap changes.

    python bench_marker_sets.py <archive> <marker_dir> 400 8

| marker set | models | SCPs/genome | HMM search | index | pairs/s/thread | median AAI |
|---|---|---|---|---|---|---|
| fastaai_122 | 122 | 78 | 117 s | 23.2 MB | 379,968 | 44.38 |
| **bac120** | 120 | **119** | 288 s | 59.7 MB | 124,814 | 44.02 |
| ar53 | 53 | 7 | 73 s | 2.5 MB | 2,664,331 | 45.96 |

## The engine is genuinely SCP-agnostic

Nothing changed but the `--hmm` file. The accession list, k-mer sets, index,
kernel and output all followed, and each set produced its own fingerprint, so
databases built on different sets refuse to be compared.

## The marker sets agree on AAI

Same genome pairs, different markers:

| | median Δ AAI | IQR | r |
|---|---|---|---|
| bac120 vs fastaai_122 | −0.46 | 0.57 | **0.9926** |
| ar53 vs fastaai_122 | +1.56 | 1.41 | 0.9729 |

**bac120 reproduces the 122-set's answers to within half an AAI point.** That is
the result the GTDB direction rests on: switching to GTDB's markers does not move
the numbers, so published FastAAI values stay interpretable.

**ar53 is the control, and it is the more surprising row.** These are bacteria,
so the archaeal set recovers 7 markers out of 53 — and still correlates at
r = 0.97. AAI is robust to marker choice because single-copy ribosomal proteins
are conserved; the choice changes precision, not the estimate.

## What a richer marker set costs

bac120 recovers 119 markers per genome against 78, because the FastAAI 122 set
is a combined bacterial+archaeal set of which only ~71 are bacterial. More
markers is more signal, and it is not free:

- **HMM search 2.5× slower** — more models match, and bac120's are longer
  (median length 298)
- **index 2.6× larger** — more SCPs per genome is more k-mers
- **search 3× slower per thread** — every pair crosses more populated accessions

None of that is a defect. It is the cost of the extra markers, and it is paid
once at build time for the index and per query for the search. Whether it is
worth it depends on whether 119 markers buys resolution that 78 does not, which
these data do not answer — the AAI values agree, so on this evidence the extra
markers mostly buy confidence rather than a different answer.

## Provenance, and why this is not GTDB's own file

`fetch_gtdb_markers.py` rebuilds both sets from primary sources — GTDB publishes
the marker *lists*, and every marker is a Pfam or TIGRFAM family available
separately, so the sets come from ~15 MB of traffic rather than the ~100 GB
GTDB-Tk package.

**GTDB pins Pfam versions and the current Pfam release has moved on.** R232 asks
for `PF00380.20`; InterPro serves `.26`. All 6 bac120 Pfams and all 12 ar53 Pfams
are at later versions; the 114 and 41 TIGRFAMs are unpinned by GTDB, so those
carry NCBI's own `.1` suffix and are not drift. Recorded per marker in
`<set>_versions.tsv`.

So these files are *GTDB's marker list at current model versions*, not GTDB's
models. Good enough to benchmark the engine; not good enough to reproduce GTDB's
trees, and any database built from them will fingerprint differently from one
built on GTDB's actual files. Getting exact versions means a versioned Pfam
release, which is a ~1.5 GB download.

TIGRFAMs now live inside NCBI's PGAP library; the JCVI mirror is gone. Models are
named by versioned accession there, so `TIGR00006` 404s and `TIGR00006.1` works —
the mapping is only in the 13 MB `hmm_PGAP.tsv`.
