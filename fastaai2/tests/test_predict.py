"""Translation-table selection.

These guard a bug that produced entirely plausible output: without a switching
margin, table 4 won on 72.8% of 2,943 Firmicutes genomes because reassigning UGA
from stop to tryptophan almost always nudges coding density up. Gene calls, SCP
sets and every AAI downstream were wrong, and nothing in the results looked odd.
"""

import pytest

from fastaai.predict import (
    BREAKER,
    DEFAULT_TABLES,
    TABLE_SWITCH_MARGIN,
    build_training_sequence,
    select_table,
)


def test_table_11_is_tried_first():
    """Order is load-bearing: the first table is the incumbent."""
    assert DEFAULT_TABLES[0] == 11


def test_marginal_improvement_does_not_switch():
    """The exact failure mode. Table 4 edges ahead on density; it must not win."""
    assert select_table([(11, 0.880), (4, 0.895)]) == 11
    assert select_table([(11, 0.880), (4, 0.900)]) == 11  # +2.3%, below margin


def test_substantial_improvement_does_switch():
    """A genuine mycoplasma should still be called as table 4."""
    assert select_table([(11, 0.500), (4, 0.900)]) == 4


def test_margin_boundary_is_strict():
    base = 0.800
    assert select_table([(11, base), (4, base * TABLE_SWITCH_MARGIN)]) == 11
    assert select_table([(11, base), (4, base * TABLE_SWITCH_MARGIN * 1.001)]) == 4


def test_worse_alternative_never_wins():
    assert select_table([(11, 0.900), (4, 0.400)]) == 11


def test_single_candidate_wins_by_default():
    assert select_table([(11, 0.1)]) == 11
    assert select_table([]) is None


def test_incumbent_is_the_first_entry_not_the_lowest_table_number():
    """Selection follows the given order, so callers control priority."""
    assert select_table([(4, 0.880), (11, 0.895)]) == 4


# ------------------------------------------------------- training sequence

def test_single_contig_gets_no_breaker():
    assert build_training_sequence([("c1", "ACGT")]) == "ACGT"


def test_multiple_contigs_are_separated_and_trailed():
    """v1 appends a breaker after the last contig too (fastaai.py:752)."""
    got = build_training_sequence([("c1", "AAAA"), ("c2", "CCCC")])
    assert got == "AAAA" + BREAKER + "CCCC" + BREAKER
    assert got.count(BREAKER) == 2


def test_empty_input():
    assert build_training_sequence([]) == ""
