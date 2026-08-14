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

## Aggregate agreement is the wrong measure

bac120 reproduces the 122-set's AAI to within half a point (median Δ −0.46,
r = 0.9926), and even ar53 manages r = 0.97. But median AAI here is 44%, so
almost every pair is unrelated and the correlation is mostly both sets agreeing
that unrelated genomes are unrelated. It says nothing about whether the *top hit*
is stable, which is what FastAAI is actually for.

The measures below are on the full 2,943 genomes, scored against GTDB R232's own
taxonomy (2,920 matched), and only on genomes that have a relative of the
relevant rank present — failing to find one that is absent is not an error.
Produced by `nearest_neighbours.py` and `worst_misses.py`; full tables in
`retrieval_results.md`.

## Retrieval is saturated, whichever markers you use

Same-genus recall at N candidates (2,395 scorable):

| marker set | markers recovered | top 1 | top 5 | top 10 |
|---|---|---|---|---|
| fastaai_122 | 78 | 99.04% | 100% | 100% |
| bac120 | 119 | 99.16% | 100% | 100% |
| ar53 | **7** | 98.79% | 99.87% | 100% |

**ar53 is the result that settles it.** These are bacteria, so the archaeal set
recovers seven markers — and retrieves a same-genus genome as the top hit 98.8%
of the time. Seven conserved markers are enough to route correctly. Going from 78
to 119 cannot improve on that because there is nothing left to win.

Head to head on the rank of the first correct hit:

| | bac120 better | fastaai_122 better | tied |
|---|---|---|---|
| genus (2,395) | 4 | 3 | 2,388 |
| family (2,837) | 7 | 6 | 2,824 |

**Net +1 genome at genus and +1 at family.** bac120's extra 41 markers buy no
measurable retrieval accuracy on this dataset.

## The sets disagree on *which* neighbour, and it does not matter

bac120 picks a different top hit than fastaai_122 for 11.1% of scorable genomes
(ar53: 25%). Of those disagreements:

| | | |
|---|---|---|
| both picks are same-genus — benign reordering | 261 | **97.8%** |
| exactly one correct | 5 | 1.9% |
| neither correct | 1 | 0.4% |

The sets shuffle the order among equally-valid congeners. The *set* of near
neighbours is stable even though the ranking within it is not, which is what a
candidate-reduction step needs.

## The worst misses are sister-family calls, not weak genomes

Rank of the first correct hit, `fastaai_122`:

| rank | P50 | P99 | P99.9 | max | misses beyond top 5 |
|---|---|---|---|---|---|
| species | 0 | 0 | 1 | 1 | 0 |
| genus | 0 | 0 | 2 | 4 | 0 |
| family | 0 | 1 | 10 | **63** | 6 (0.21%) |

A top-100 shortlist captures every correct answer in this dataset, and genus
needs only top-5.

The six family misses carry a **median 78 SCPs — identical to the population**,
so they are not marker-poor genomes that a wider shortlist would rescue. Every
one is an adjacent-family confusion in the 50–62% AAI band:

    Streptococcaceae -> Enterococcaceae   58.68     (both Lactobacillales)
    Vagococcaceae    -> Enterococcaceae   62.05
    Halarsenatibac.  -> Halanaerobiac.    50.68
    UBA6769          -> Guptibacillaceae  59.16

Half involve GTDB placeholder families (`YIM-B00363`, `NBRC-103111`, `UBA6769`)
where the boundary is provisional, so the truth label is itself soft. bac120
trades on these rather than fixing them — it improves four and worsens two.

## What a richer marker set costs

bac120 recovers 119 markers per genome against 78, because the FastAAI 122 set
is a combined bacterial+archaeal set of which only ~71 are bacterial. More
markers is more signal, and it is not free:

- **HMM search 2.5× slower** — more models match, and bac120's are longer
  (median length 298)
- **index 2.6× larger** — more SCPs per genome is more k-mers
- **search 3× slower per thread** — every pair crosses more populated accessions

None of that is a defect — it is the cost of carrying the extra markers. But the
retrieval results above show those markers buy no accuracy here, so on this
dataset the cost is unrecovered.

**The case for bac120 is not accuracy — it is shared preprocessing.** GTDB-Tk's
`identify` step already runs Prodigal and hmmsearch against exactly these
markers. A pipeline that classifies with GTDB-Tk and reduces candidates with
FastAAI pays that stage once instead of twice, and FastAAI's own cost collapses
to k-merisation and search, which is seconds. Read that way the 2.5x HMM-search
penalty is not a cost FastAAI adds; it is a cost the pipeline is already paying,
and FastAAI riding along on it is free.

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
