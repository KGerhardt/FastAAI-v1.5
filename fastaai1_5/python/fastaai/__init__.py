"""FastAAI 1.5 — average amino acid identity from single-copy protein tetramers.

Python owns FASTA ingestion, gene prediction and HMM search; the Rust extension
owns k-merisation, the inverted index, counting and the AAI transform.
"""

from . import _core
from .ingest import find_genomes, genome_name, read_fasta
from .reshape import per_genome as results_per_genome
from .reshape import to_matrix as results_to_matrix
from .api import (
    all_steps,
    build_database,
    describe_database,
    dump_database,
    genome_to_protein,
    genomes_to_proteins,
    preprocess,
    prot_hmm_to_crystal,
    protein_to_hmm,
    proteins_to_hmms,
    read_hit_table,
)
from .pipeline import (
    DEFAULT_SEARCH_THREADS,
    GenomeRecord,
    Match,
    SearchResult,
    build_from_crystals,
    crystallize_archive,
    preprocess_one,
    preprocess_paths,
    search,
)
from .predict import predict_proteins
from .search import DEFAULT_FILTER, ModelSet, best_hits

__version__ = "1.5.0"

Database = _core.Database
compare_pair = _core.compare_pair
open_database = _core.open_database
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
    "Match",
    "MAX_PARTITION",
    "ModelSet",
    "SearchResult",
    "aai_to_jaccard",
    "best_hits",
    "compare_pair",
    "find_genomes",
    "open_database",
    "genome_name",
    "jaccard_to_aai",
    "kmerize",
    "predict_proteins",
    "all_steps",
    "build_database",
    "describe_database",
    "dump_database",
    "build_from_crystals",
    "crystallize_archive",
    "genome_to_protein",
    "genomes_to_proteins",
    "preprocess",
    "preprocess_one",
    "preprocess_paths",
    "prot_hmm_to_crystal",
    "protein_to_hmm",
    "proteins_to_hmms",
    "read_hit_table",
    "results_per_genome",
    "results_to_matrix",
    "read_fasta",
    "search",
    "__version__",
]
