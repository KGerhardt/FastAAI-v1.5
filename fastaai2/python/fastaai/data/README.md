# Bundled model set

`Complete_SCG_DB.hmm.gz` — the 122 single-copy protein HMMs FastAAI 1 shipped,
byte-identical to `FastAAI/fastaai/00.Libraries/01.SCG_HMMs/Complete_SCG_DB.hmm`
and gzipped (9.2 MB -> 2.0 MB; pyhmmer reads the stream directly).

This is the default so that an install works without hunting for models, and so
that databases built with defaults share one fingerprint and are mutually
comparable. It is not compiled in: `--hmm` overrides it, the accession list comes
from whichever file is used, and a database records which model set built it.
