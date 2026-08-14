# FastAAI 1.5

Average amino acid identity between microbial genomes, estimated from tetramer
sketches of single-copy protein-coding genes.

This is FastAAI with its search engine rewritten in Rust. Same algorithm, but faster and lighter weight.

```
FASTA ──► pyrodigal ──► pyhmmer ──► best-hit filter ──► k-merise ──► inverted index
          └──────────── Python ────────────┘          └────────── Rust ──────────┘
```

Python manages gene prediction and HMM search via Cython bindings to prodigal (pyrodigal) and HMMER (pyHMMER). Rust replaces a numpy implementation of the FastAAI search from FastAAI v1.

The search engine differences are (1) searches are partitioned to batches of no more than 16,384 genomes at a time, limiting the maximum memory required to execute a search regardless of database size. (2) The original search approach calculated the similarity of one query genome vs. all search targets iteratively; v1.5 compares batches of genomes to batches of genome in an all-vs-all search, which eliminates repetitive lookup and aggregation operations. A single genome query is just a case where batch size = 1 on the query side. (3) Outputs are now partitioned to match the partition structure of the database. Output TSVs and matrices will never be more than 16384x16384 genomes at a time. Output structure is identical. These could be reshaped into v1's intact outputs.

Some additional conveniences have been added, primarily in the form of an API for Python. The API exposes a preprocessing workflow for genomes -> FastAAI database and search of database vs. database, plus some operators for interacting with FastAAI results. Slightly more granular equivalents of the preprocessing steps are also available.

## Install

```sh
cd fastaai1_5
maturin develop --release      # or: pip install .
```

## Usage: CLI

Two verbs do the work, and three more operate on what they left behind:

```sh
fastaai build   inputs                   # genomes or proteins -> a database
fastaai query   -q database [-t other]   # database x database -> AAI

fastaai crystallize  source              # stored proteins + hits -> crystals
fastaai inspect      database            # a database -> readable text
fastaai reshape      results             # block output -> v1's output shapes
```

With no options at all, a build and a self-search are two lines:

```sh
fastaai build /path/to/genomes
fastaai query -q FastAAI/database
```

Everything they produce lands under one root, `FastAAI/` in the working
directory. Nothing is discarded and nothing is written outside it:

```text
FastAAI/
├─ proteins/<genome>.fasta            every gene call
├─ hmm_hits/<genome>.tsv              every raw HMM hit
├─ crystals/<genome>.crystal.fasta    the SCPs that won their accession
├─ database/                          schema, manifest, part.00000 ...
├─ results/                           block_q00000_t00000.tsv ...   (query adds this)
├─ models.txt                         the accessions searched for, in order
└─ models.sha256                      their fingerprint, which the database records too
```

`FastAAI/database` **is** the database - there is no name level beneath it - so
that path is what you hand back to `query`. `--dir` moves the whole root
somewhere else, and `-d` is only needed to keep more than one database under a
single root.

```sh
# proteins rather than genomes; gene prediction is skipped
fastaai build /path/to/proteins --input protein

# another root, gzip every file written, 16 preprocessing workers
fastaai build /path/to/genomes --dir /scratch/run17 --gzip --processes 16

# a different marker set: an HMM file, or a packaged set by name
fastaai build /path/to/genomes --hmm my_markers.hmm
fastaai build /path/to/genomes --hmm gtdb-bact

# rebuild straight from crystals - no prediction, no HMM search
fastaai build FastAAI/crystals

# two databases, and an explicit output path
fastaai query -q FastAAI/database -t /scratch/run17/database -o aai.tsv
fastaai query -q FastAAI/database -o -                    # to stdout
fastaai query -q FastAAI/database --output_style matrix

# resolve crystals from proteins and hits already on disk
fastaai crystallize old_run/

# the packed binary database, written back out as text
fastaai inspect FastAAI/database

# blocks back into the shapes v1 wrote
fastaai reshape FastAAI/results --per-genome per_genome/ --matrix aai.matrix
```

### Options

Both `build` and `query` take all of these, since `query` accepts raw sequences
and will preprocess them the same way `build` does. The last two are search
options and do nothing on a `build`:

| flag | default | |
|---|---|---|
| `--hmm FILE\|SET` | the bundled 122-SCP set | an HMM file, plain or gzipped, or a packaged set by name |
| `--dir DIR` | `FastAAI/` here | output root, created if needed |
| `--gzip` | off | gzip each file as it is written |
| `--processes N` | 4 | worker processes for preprocessing - prediction, HMM search, crystallising |
| `--filter v1\|v1_alt\|rbh` | `v1` | best-hit resolution |
| `--input auto\|genome\|protein` | `auto` | input type; `protein` skips gene prediction |
| `--limit N` | all | use only the first N inputs |
| `--quiet` | off | suppress progress on stderr |
| `--threads N` | 8 | threads for the search kernel |
| `--do_stdev` | off | also report the standard deviation of Jaccard across shared SCPs |

`fastaai build INPUTS`:

| flag | default | |
|---|---|---|
| `-d`, `--database` | `<dir>/database/` | a name adds levels beneath the root; an absolute path leaves it |
| `--source LABEL` | none | provenance recorded in the database, e.g. `'GTDB R232 bac120'` |

`fastaai query`:

| flag | default | |
|---|---|---|
| `-q`, `--query` | *required* | a database, an archive, a crystal directory, or sequences |
| `-t`, `--target` | the query itself | a self-search takes the symmetric upper-triangle path |
| `-o`, `--output` | `<dir>/results/` | a directory of block files; `-` is stdout, for a single-block search |
| `--output_style tsv\|matrix` | `tsv` | matrix holds AAI only |
| `--emit both\|aai\|jaccard` | `both` | `aai` drops `avg_jacc_sim`, `jaccard` drops `AAI_estimate` |

`fastaai crystallize SOURCE`, where *source* is a root holding `proteins/` and
`hmm_hits/`:

| flag | default | |
|---|---|---|
| `-o`, `--output` | `<source>/crystals/` | |
| `--hmm FILE\|SET` | the bundled set | the models the archive was searched with |
| `--filter v1\|v1_alt\|rbh` | `v1` | best-hit resolution baked into the crystals |
| `--processes N` | 1 | scales 5-6x at 8 |
| `--gzip`, `--quiet` | off | |

`fastaai inspect DATABASE`:

| flag | default | |
|---|---|---|
| `-o`, `--output` | `<database>/../inspect/` | |
| `--by genome\|kmer\|both` | `both` | `genome`: what each genome contains, which lines up with its crystal. `kmer`: the index as stored |
| `--full` | off | list every member rather than a count, reconstructing the index exactly. Large - tens of millions of rows at GTDB scale |
| `--quiet` | off | |

`fastaai reshape RESULTS`, which needs at least one of the two outputs:

| flag | default | |
|---|---|---|
| `--per-genome DIR` | - | one TSV per query genome, that genome against everything |
| `--matrix FILE` | - | the AAI-only matrix |
| `--gzip`, `--quiet` | off | |

## Usage: Python API

```python
import fastaai

# Any rank of input to a database in one call. Ranks combine.
db = fastaai.preprocess(genomes="/path/to/genomes")
db = fastaai.preprocess(proteins=["a.faa", "b.faa"], crystals="other/crystals")

res = fastaai.search(db, db, threads=8)
```

`preprocess` fills the same root the CLI does - `FastAAI/` unless `directory=`
says otherwise - and writes the database to `FastAAI/database/`. `database=`
names something beneath that root or gives an absolute path, and `save=False`
builds without writing one. Its other arguments are `PH_tups=` for
`(protein, hmm)` pairs, plus `models=`, `filter_mode=`, `processes=` and
`compress=`, which mean what `--hmm`, `--filter`, `--processes` and `--gzip`
mean on the command line.

Reading the result. The matrices stay public, but the questions people actually
ask have answers:

```python
res.queries, res.targets     # the genomes on each side, in row/column order
res.shape                    # (n_queries, n_targets)
res.scps("GCF_000007085.1")  # how many markers that genome carries

res.best_hit("GCF_000007085.1")        # Match(query, target, aai, jaccard, shared, poss_shared)
res.hits_for("GCF_000007085.1", k=5)   # its five nearest, best first
res.top_hits(k=5)                      # that, for every query

# Filtered iteration. Call it to filter, or iterate it for everything.
for m in res(query="any", min_aai=60, min_shared_frac=0.5):
    print(m.query, m.target, m.aai, m.shared_frac)

res.jaccard   # (n, n) float64, NaN where no accession is shared
res.shared    # (n, n) uint32, accessions carried by both genomes
res.aai       # (n, n) float64, uncensored
```

`query=` and `target=` take `"any"` for all of them, one name, or a collection
of names; the thresholds are inclusive. `min_shared_frac` is
`shared / poss_shared` - of the markers the poorer genome carries, the fraction
actually compared, which is what separates a genuinely distant pair from a pair
where one genome is a bad assembly.

A self-comparison's diagonal is a genome against itself, so it is **not** a
neighbour: `hits_for` and `best_hit` skip it unless you pass
`include_self=True`. Pairs sharing no marker are dropped rather than reported as
zero - no shared marker is an absence of evidence, not evidence of distance.

`res.to_tsv(path)` writes FastAAI 1's table - same columns, same names, same
order. The band and the rounding are the engine's own, exposed rather than
reimplemented, and the output is asserted byte-for-byte against
`Database.write_block`; two writers for one format is how the two drift. A
search too large to hold in memory writes its blocks straight from Rust
instead.

Or one step at a time. Each step comes in two forms - a **unit** that does one
genome and returns a path, and a **driver** that runs it over many in parallel:

```python
prots = fastaai.genomes_to_proteins(genomes, "FastAAI/proteins", processes=8)
hits  = fastaai.proteins_to_hmms(prots, "FastAAI/hmm_hits", processes=8)
cry   = fastaai.prot_hmm_to_crystal(zip(prots, hits), "FastAAI/crystals", processes=8)

db    = fastaai.build_database("FastAAI/crystals", save_to="FastAAI/database")
res   = fastaai.search(db, db, threads=8)
```

`all_steps(genome, directory)` is the whole chain for one genome - predict,
search, resolve, write - with nothing returned between stages, and `preprocess`
is its driver. That is the shape the parallelism wants: one worker owns one
genome from FASTA to crystal, so no intermediate crosses a process boundary and
there is no collector to funnel through. Running the three steps as three
parallel passes would be the same work with two extra synchronisation points
and the intermediates read back off disk.

The two reshaping helpers are the `fastaai reshape` verb, importable:

```python
fastaai.results_per_genome("FastAAI/results", "per_genome/")  # one TSV per query genome
fastaai.results_to_matrix("FastAAI/results", "aai.matrix")    # the AAI-only matrix
```

Both stream, so peak memory is one row rather than one matrix.
`fastaai.describe_database(db)` and `fastaai.dump_database(db, out)` are
`inspect`, likewise.

## Output format

**The TSV is FastAAI 1's, unchanged** - same columns, same names, same order, so
a parser written against v1 keeps working:

```
query  target  avg_jacc_sim  jacc_SD  num_shared_SCPs  poss_shared_SCPs  AAI_estimate
```

Numbers are rendered as `str(numpy.round(v, dp))`, matching v1 digit for digit:
Jaccard and its standard deviation to 4 decimal places, AAI to 2.

| | |
|---|---|
| `jacc_SD` | always present; reads `N/A` unless `--do_stdev` was given |
| `poss_shared_SCPs` | `min(query SCPs, target SCPs)` - a pair cannot share more markers than the poorer genome carries |
| `AAI_estimate` | `<30%` / `>90%` outside the regression's sensitivity band; `100.0` on the diagonal of a self-search |
| a pair sharing no marker | `N/A` in every value column |

**`--output_style matrix`** writes a Q×T grid of AAI - one row per query, one
column per target, `query_genome` in the corner - with v1's `15.0` and `95.0`
standing in for the two categorical labels, since a cell cannot hold a string.

**The diagonal of a self-search reads `100`** - `100.0` in both the matrix and
the TSV's `AAI_estimate`. Identity there is given by the comparison, not inferred
from it; the regression is fitted and unbounded above, so consulting it returns a
value past 100 that reports as the `>90%` sentinel, which is uncertainty about
something that is not uncertain.

Nothing else is exempt, and the exemption is a property of the *comparison*, not
of the genome. Two distinct genomes that happen to be identical are a
measurement that came out at the ceiling - equality of content is not identity.
So is the same genome held in two different databases and compared across them:
the engine is not told they are one genome, so that pair reads `95.0` in the
matrix and `>90%` in the TSV like any other ceiling measurement.

Two deliberate departures from v1: (1) a pair sharing no marker is `N/A` in the
matrix where v1 writes `0`, which cannot be differentiated from a real
measurement of zero, and (2) the diagonal of a self-search reports `100`
where v1 reports `>90%`, to indicate genuine identity.

### Blocks

v1 output TSV files either as a monolith (all queries vs. all targets) or as individual, per-query-genome
TSVs that were each one-to-all. It also supported a monolithic matrix output of just AAI values.

v1.5 writes one file per (query partition × target partition) block:

```text
                     target partitions
                 t00000     t00001     t00002
               ┌──────────┬──────────┬──────────┐
        q00000 │   file   │   file   │   file   │
               ├──────────┼──────────┼──────────┤
        q00001 │   file   │   file   │   file   │   ← query partitions
               ├──────────┼──────────┼──────────┤
        q00002 │   file   │   file   │   file   │
               └──────────┴──────────┴──────────┘

  FastAAI/results/
  ├─ block_q00000_t00000.tsv
  ├─ block_q00000_t00001.tsv
  ├─ …
  └─ block_q00002_t00002.tsv
```

This is true for both output formats; `--output_style matrix` writes the same
grid of files with a `.matrix` extension. Every TSV block carries the header, so
blocks concatenate into one valid v1 table. The
Python API includes search functions to extract single genome information, and
there are helper functions to produce the same per-genome one-vs-all TSVs as
before, as well as converting TSVs to a matrix (the reverse direction is not
possible because the matrices hold only a subset of information.)

## Preprocessing outputs

The same tree as above, with what each rank costs and where it comes from:

```text
  genome.fna.gz  (yours, named however you like)
       │
       │  pyrodigal ····································   1.79 s   64%
       ▼
   every protein ─────────────────►  <root>/proteins/<genome>.fasta<.gz>
       │
       │  pyhmmer ······································   1.03 s   36%
       ▼
   every raw hit ─────────────────►  <root>/hmm_hits/<genome>.tsv<.gz>
       │
       │  best-hit filter ······························    ~0 s
       ▼
   the SCPs that won ─────────────►  <root>/crystals/<genome>.crystal.fasta<.gz>
       │
       │  then Rust, once, over the whole crystal set
       ▼
   k-merise, invert, partition ───►  <root>/database/
                                     5 s for 2,943 genomes

   a search adds ─────────────────►  <root>/results/block_qNNNNN_tNNNNN.tsv
```

The database is packed binary. `fastaai inspect <database>` writes it back out
as text - the schema, the genome and accession tables, and the index in either
direction: `by_genome.tsv` for what a genome contains, `by_kmer.json` for which
genomes share a k-mer, which is the form it is actually stored in.

Each rank is a place to re-enter the pipeline, which is the point of storing
them:

| start from | skips | per genome |
|---|---|---|
| genomes | nothing | ~2.8 s |
| proteins (`--input protein`) | prediction | ~1.0 s |
| stored proteins + hits | prediction and search | re-resolve only |
| crystals | everything | **~1.7 ms** |

Measured over all 2,943 Firmicutes. Files are plain text unless `--gzip` is
given, so both are shown:

| rank | per genome | 2,943 Firmicutes | with `--gzip` | survives a change of |
|---|---|---|---|---|
| proteins | 1,074 KB | 3,237 MB | 1,794 MB | model set and filter - anything |
| raw HMM hits | 3.4 KB | 10.1 MB | 2.8 MB | filter only; they name a model set |
| crystals | 28.6 KB | 86.1 MB | 31.9 MB | nothing, but rebuilds in 5 s |
| the database | 38.7 KB | 116.5 MB | 63.0 MB tar.gz | (the built artifact) |

**Crystals are the resolved SCPs** - one FASTA per genome holding just
the marker proteins that won their accession, each record labelled with the
genome, the originating gene call, the model-set fingerprint and the filter that
produced it. These are exactly and only the information that FastAAI uses to compute AAI.

They are also *how* a database gets built. Each preprocessing worker writes its
own crystal and the build step reads them sequentially. Peak memory stops tracking 
the size of the collection as a result: a worker drops a genome's sequences once 
its crystal is written, and the build streams them one file at a time.

```sh
fastaai build genomes/                    # crystals written to FastAAI/crystals/
fastaai crystallize old_run/              # or from proteins and hits you already have
fastaai build FastAAI/crystals            # rebuild, no prediction or search
```

When crystalized, the 2,943 Firmicutes genomes are ~86MB unzipped and ~32MB zipped.
Rebuilding a 112 MB FastAAI database from it takes **5 s** - against
roughly two CPU-hours to preprocess the same genomes. All 8,661,250 pairs agree
with the database built directly from the stored hits. The database is 63MB zipped.

If you're going to share the contents of a large database build effort, it would be wisest
to share the crystals in a tarball instead of directly shipping the database. They're more
compact, they're more transparent, and you can add more genomes to a collection. You can't do 
that with a database in v1.5

## Marker sets

**The 122 SCPs FastAAI 1 shipped are bundled and used by default**, so an install
works without hunting for models. They are a default, not a fixture: `--hmm` takes
any HMM file, plain or gzipped, and the accession list, index and output all
follow from it. Every database records which model set built it, so defaults and
overrides cannot be mixed by accident.

GTDB's marker sets are packaged too, reachable by name because a file inside
site-packages is not a path anyone wants to type:

| `--hmm` | models | |
|---|---|---|
| *(omitted)* | 122 | FastAAI 1's SCPs - the default |
| `gtdb-bact` | 120 | GTDB bac120, bacteria |
| `gtdb-arch` | 53 | GTDB ar53, archaea |
| `gtdb-all` | 168 | the union of both - they share 5 markers, so this is not 173 |

Case and underscores are interchangeable (`GTDB_BACT` works). These are
assembled from Pfam and TIGRFAM rather than copied from GTDB-Tk, and the pinned
Pfam versions have since moved on, so they benchmark the engine but **do not
reproduce GTDB's trees** - see `python/fastaai/data/README.md`.

The three sets agree closely enough that there is no accuracy reason to prefer
one over another; choose on which markers your other tooling already produces.
See `fastaai1_5/methods/marker_sets.md` for the comparison.

## Database representation

FastAAI v1 used a SQLite3 database to store SCP kmer data for each genome. There are two
representations: genome-first, and kmer-first. The genome-first representation encodes 
the per-SCP kmer list for each SCP in each genome. It's essentially the data in a crystal.
The kmer-first representation stores a list of kmers for each SCP, with a list of genomes
that (A) had a copy of that SCP and (B) contained that particular kmer in their copy.

The kmer-first representation enables the rapid calculation of set intersections for 
Jaccard distances: go get the list of target genomes for each kmer in a query genome's 
copy of a particular SCP, then count the number of times you see each target genome. A 
little set math to get the union, and Jaccard falls out.

```text
FastAAI 1                                   one SQLite file
fastaai.db
├─ genome_index                             name → id, protein count
├─ genome_acc_kmer_counts                   (genome, accession) → count
├─ PF00380                    ── inverted   kmer   → genomes[]
├─ PF00380_genomes            ── forward    genome → kmers[]
├─ PF00410  /  PF00410_genomes
└─ …                                        2 tables per accession,
                                            244 tables for 122 SCPs

adding genomes:  INSERT … ON CONFLICT DO UPDATE SET genomes = genomes || (?)
                 read-modify-write of every posting list the genome touches
```

As FastAAI v1 was all Python, parallelism options were limited. In a search, each worker would
use a single genomes' genome-first representation (either in the same or a different database) 
to query the kmer-first representation, from which it would calculate AAI for that query
genome against all target genomes. This prevents memory blowup, but does result in many, many
SQL queries and redundant retrievals and reprocessing of a great deal of database contents.

v1.5 keeps a directory, stores only the kmer-first representation, and cuts it into
partitions that are independent of one another:

```text
FastAAI 1.5                                 a directory
FastAAI/database/
├─ schema                                   k, alphabet, ordered accessions,
│                                           filter mode, model fingerprint
├─ manifest                                 genome → ordinal, partition,
│                                           local id, hash, name
├─ part.00000  ┐                            CSR per accession, direct-addressed
├─ part.00001  │  inverted index only       by k-mer id:
└─ part.NNNNN  ┘                            offsets[kmer]..offsets[kmer+1]
                                            slices postings
                                            ≤16,384 genomes, u16 local ids,
                                            sorted ascending

adding genomes:  Not supported, add a new crystal and rebuild the database from crystals.
```

The use of partitioning enables FastAAI v1.5 to perform a kmer-kmer search across two kmer-first
representations while still using reasonable RAM allocations (a max-sized partition search with std.
deviation calculation on takes ~1GB RAM). This eliminates all redundant lookups of kmer information
and enables the quicker tabulation of results that accelerates v1.5 relative to v1.

## Performance

**Search**, 8 threads. v1 was run through `db_query --in_memory --store_results`, its
fastest path:

This was done on my laptop, so don't use the numbers as a final estimate of performance. The point is that it's quite a bit faster than v1. Throughput is pairwise genome comparisons per second per thread.

| scale | pairs | v1 | v1.5 | v1 /s/thread | v1.5 /s/thread |
|---|---|---|---|---|---|
| **2,943 genomes** | **8,661,249** | **21.84 s** | **1.587 s** | **49,572** | **682,203** |

Memory and disk at 2,943 genomes:

| | v1 | v1.5 |
|---|---|---|
| peak RSS | 1.01 GB | 116 MB index |
| database on disk | 508 MB | 116 MB |

**Preprocessing costs**

| stage | median s/genome | share |
|---|---|---|
| predict (pyrodigal) | 1.79 | 64% |
| hmmsearch (pyhmmer) | 1.03 | 36% |

## Agreement with FastAAI 1

FastAAI v1.5 has a small bugfix to the calculation of Jaccard similarity between genomes. v1 
admits the stop symbol `*` and ambiguous residues `X`, as parts of valid kmers for comparison, 
while v1.5 does not. The bug produced tiny errors that wouldn't affect the material 
interpretation of any results, but it's fixed nonetheless.

120 Firmicutes genomes, all-vs-all, against FastAAI 1 driven through its own
`aai_index` module.

![v1 vs v1.5 concordance](fastaai1_5/methods/concordance_v1_v15.png)

| | |
|---|---|
| pairs compared (off-diagonal) | 14,280 |
| shared-SCP differences | 0 |
| median \|Δ AAI\| | 0.0115 percentage points |
| max \|Δ AAI\| | 0.0627 percentage points |

See **[`fastaai1_5/methods/`](fastaai1_5/methods/)** for the data and the harness for this comparison.

**FastAAI 1 command lines still work.** `build_db`, `db_query`, `aai_index`,
`single_query`, `multi_query` and `simple_query` are rerouted to the new verbs,
with arguments preserved where they still mean something and a diagnostic where
they do not. `merge_db` is the one that no longer has a target: it exits saying
so and gives the crystal-and-rebuild replacement.

## Status

Working end to end: on-disk partitioned databases, the three stored
preprocessing ranks, crystal-driven builds, the FastAAI 1 compatible CLI, and
optional per-pair Jaccard standard deviation (`--do_stdev`). 291 Python and 53
Rust tests.

Not yet packaged for bioconda.

## Dependencies

`pyrodigal`, `pyhmmer`, `pyfastx`, `numpy`. 

**numpy 1.x and 2.x both work, and produce identical output.** The 2.0 removals
(`np.float_` and friends) split a lot of downstream code; this package restricts
itself to spellings valid under both, and `tests/test_numpy_compat.py` fails if
one of the removed aliases is reintroduced. The suite passes under 1.26.4 and
2.4.6, and a query gives byte-identical TSV, matrix and API results under each.

numpy is used in one place - `SearchResult`, the in-memory Python API. The CLI
does not touch it; formatting and output are Rust.

## Licence

GPL-3.0-or-later - see [LICENSE](LICENSE). This is not only a preference:
FastAAI 1.5 imports pyrodigal, which is GPL-3.0-or-later, so a compatible
licence is required rather than chosen.

The bundled HMMs in `fastaai1_5/python/fastaai/data/` are redistributed from
FastAAI 1 under the MIT licence and keep their original notice; see the README
beside them. FastAAI 1 itself remains at
https://github.com/cruizperez/FastAAI.
