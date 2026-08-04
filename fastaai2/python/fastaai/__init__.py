"""FastAAI 2 — average amino acid identity from single-copy protein tetramers.

Python owns FASTA ingestion, gene prediction and HMM search; the Rust extension
owns k-merisation, the inverted index, counting and the AAI transform.
"""

from . import _core
from .ingest import find_genomes, genome_name, read_fasta
from .pipeline import (
    DEFAULT_SEARCH_THREADS,
    GenomeRecord,
    SearchResult,
    build_database,
    preprocess,
    preprocess_one,
    search,
)
from .predict import predict_proteins
from .search import DEFAULT_FILTER, ModelSet, best_hits

__version__ = "2.0.0a1"

Database = _core.Database
jaccard_to_aai = _core.jaccard_to_aai
aai_to_jaccard = _core.aai_to_jaccard
kmerize = _core.kmerize
DEFAULT_ALPHABET = _core.DEFAULT_ALPHABET
DEFAULT_K = _core.DEFAULT_K
MAX_PARTITION = _core.MAX_PARTITION

__all__ = [
    "Database",
    "DEFAULT_ALPHABET",
    "DEFAULT_FILTER",
    "DEFAULT_K",
    "DEFAULT_SEARCH_THREADS",
    "GenomeRecord",
    "MAX_PARTITION",
    "ModelSet",
    "SearchResult",
    "aai_to_jaccard",
    "best_hits",
    "build_database",
    "find_genomes",
    "genome_name",
    "jaccard_to_aai",
    "kmerize",
    "predict_proteins",
    "preprocess",
    "preprocess_one",
    "read_fasta",
    "search",
    "__version__",
]
