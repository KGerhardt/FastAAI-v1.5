# Bundled model set

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
