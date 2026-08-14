# Bundled model sets

| file | models | selected by |
|---|---|---|
| `Complete_SCG_DB.hmm.gz` | 122 | the default — no `--hmm` |
| `gtdb_bac120.hmm.gz` | 120 | `--hmm gtdb-bact` |
| `gtdb_ar53.hmm.gz` | 53 | `--hmm gtdb-arch` |
| both GTDB files | 168 | `--hmm gtdb-all` |

`gtdb-all` is the *union* of the other two, assembled when it is loaded rather
than shipped as a third file. bac120 and ar53 share 5 markers (PF00410, PF00466,
TIGR00064, TIGR00967, TIGR01171), so a concatenation would repeat those
accessions — and accession IDs are positions in the model list, so a repeat
makes one position unreachable. Building it at load time also avoids shipping
4.5 MB that is 168/173 identical to files already present.

Every set fingerprints differently, so databases built from different sets
refuse to be compared.

## The default set

`Complete_SCG_DB.hmm.gz` — the 122 single-copy protein HMMs FastAAI 1 shipped,
byte-identical to `FastAAI/fastaai/00.Libraries/01.SCG_HMMs/Complete_SCG_DB.hmm`
and gzipped (9.2 MB -> 2.0 MB; pyhmmer reads the stream directly).

This is the default so that an install works without hunting for models, and so
that databases built with defaults share one fingerprint and are mutually
comparable. It is not compiled in: `--hmm` overrides it, the accession list comes
from whichever file is used, and a database records which model set built it.

## Provenance and licence

These models are redistributed from FastAAI 1
(https://github.com/cruizperez/FastAAI), which ships them under the MIT licence:

> The MIT License (MIT)
> Copyright © 2022 Kenji Gerhardt, Carlos Ruiz-Perez, Miguel Rodriguez-Rojas,
> Konstantinos Konstantinidis

MIT requires that notice travel with the files, so it is reproduced here. The
models themselves are drawn from Pfam and TIGRFAMs. FastAAI 1.5's own code is
GPL-3.0-or-later; that does not relicence these, and the notice above governs
them.

## The GTDB sets

`gtdb_bac120.hmm.gz` and `gtdb_ar53.hmm.gz` hold the marker sets GTDB uses for
its bacterial and archaeal trees, assembled by `methods/fetch_gtdb_markers.py`
and gzipped (19.3 MB -> 3.5 MB, 6.3 MB -> 1.1 MB).

**These are not GTDB's own files, and a database built from them will not
reproduce GTDB's trees.** GTDB does not distribute the HMMs separately — they
are inside the ~100 GB GTDB-Tk package — but the marker *lists* are published
and every marker is a Pfam or TIGRFAM family available on its own. The lists pin
Pfam versions that Pfam has since moved past (R232 asks PF00380.20; InterPro
serves .26), so all 18 version-pinned markers across the two sets are at later
versions. Same families, later models. The versions actually fetched are
recorded per marker in `methods/bac120_versions.tsv` and
`methods/ar53_versions.tsv`.

These models are redistributed from public reference databases, cited here so
the provenance travels with the files:

| what | from |
|---|---|
| bac120 / ar53 marker lists | GTDB, https://data.gtdb.ecogenomic.org/releases/latest/auxillary_files/ |
| Pfam models | EBI InterPro, https://www.ebi.ac.uk/interpro/ |
| TIGRFAM models | NCBI PGAP HMM library, https://ftp.ncbi.nlm.nih.gov/hmm/ |

Each is a public resource under its own terms rather than FastAAI's
GPL-3.0-or-later, which covers this repository's code and not these files. The
exact models fetched are listed per marker in `methods/bac120_versions.tsv` and
`methods/ar53_versions.tsv`, so any model here can be traced to its source
family and version.

If GTDB, Pfam or NCBI would prefer these not be redistributed, removing them is
a one-line change: drop the file from `python/fastaai/data/` and its entry from
`MODEL_SETS` in `search.py`. `--hmm` still takes any HMM file, so nothing about
the tool depends on them being bundled.

Retrieval was measured against GTDB R232 taxonomy on 2,943 genomes and is
saturated for all three sets — see `methods/retrieval_results.md`. The reason to
choose a GTDB set is shared preprocessing with GTDB-Tk, not accuracy.
