"""Overlap (Dice) validation on substituted benzenes.

Cases:
- identical molecules (benzene and chlorobenzene, each with a copy) -> overlap 1
- chlorine substituent -> overlap with benzene drops below 1
- disubstituted compounds -> share a chlorine with chlorobenzene, so overlap
  with chlorobenzene is higher than benzene's overlap with the same compound

Properties are set by hand (see molecule_factory.PROP_VALUES); no chemical
accuracy is implied.
"""

from itertools import combinations

import numpy as np
import pytest

from molecule_factory import PROPS
from quartz.dataprocessing import dice_coefficient

PARENT = "BEN"
MONO = "CLB"
DI = ["D12", "D13", "D14"]
ALL_CODES = [PARENT, MONO, *DI]

# Regression snapshot of the full Dice matrix (4 dp), produced by the current
# implementation. Not an independent reference value.
SNAPSHOT = {
    "polarizability": {
        ("BEN", "CLB"): 0.8709,
        ("BEN", "D12"): 0.7840,
        ("BEN", "D13"): 0.7790,
        ("BEN", "D14"): 0.7497,
        ("CLB", "D12"): 0.9077,
        ("CLB", "D13"): 0.8999,
        ("CLB", "D14"): 0.8699,
        ("D12", "D13"): 0.8293,
        ("D12", "D14"): 0.8042,
        ("D13", "D14"): 0.8000,
    },
    "charges": {
        ("BEN", "CLB"): 0.4062,
        ("BEN", "D12"): 0.1861,
        ("BEN", "D13"): 0.1639,
        ("BEN", "D14"): 0.1655,
        ("CLB", "D12"): 0.6907,
        ("CLB", "D13"): 0.6592,
        ("CLB", "D14"): 0.6708,
        ("D12", "D13"): 0.4788,
        ("D12", "D14"): 0.4843,
        ("D13", "D14"): 0.4638,
    },
}


def _dice(grids, a, b, prop):
    return dice_coefficient(grids[a], [grids[b]])[prop][b]


@pytest.mark.parametrize("prop", PROPS)
@pytest.mark.parametrize("original, copy", [(PARENT, "BEN_copy"), (MONO, "CLB_copy")])
def test_identical_molecules_have_unit_overlap(substituted_grids, prop, original, copy):
    assert _dice(substituted_grids, original, copy, prop) == pytest.approx(1.0)


@pytest.mark.parametrize("prop", PROPS)
@pytest.mark.parametrize("original, copy", [(PARENT, "BEN_copy"), (MONO, "CLB_copy")])
def test_identical_molecules_have_identical_grids(
    substituted_grids, prop, original, copy
):
    np.testing.assert_array_equal(
        substituted_grids[original].npGrid[prop],
        substituted_grids[copy].npGrid[prop],
    )


@pytest.mark.parametrize("prop", PROPS)
@pytest.mark.parametrize("code", [MONO, *DI])
def test_chlorine_substitution_lowers_overlap_with_benzene(
    substituted_grids, prop, code
):
    overlap = _dice(substituted_grids, PARENT, code, prop)
    assert -1.0 <= overlap < 1.0


@pytest.mark.parametrize("prop", PROPS)
@pytest.mark.parametrize("code", DI)
def test_shared_chlorine_raises_overlap_with_disubstituted(
    substituted_grids, prop, code
):
    # chlorobenzene and every dichlorobenzene carry a Cl at ring position 0
    with_shared_cl = _dice(substituted_grids, MONO, code, prop)
    benzene_only = _dice(substituted_grids, PARENT, code, prop)
    assert with_shared_cl > benzene_only


@pytest.mark.parametrize("prop", PROPS)
@pytest.mark.parametrize("a, b", list(combinations(DI, 2)))
def test_regioisomers_are_distinguishable(substituted_grids, prop, a, b):
    assert _dice(substituted_grids, a, b, prop) < 1.0


@pytest.mark.parametrize("prop", PROPS)
def test_overlap_is_symmetric(substituted_grids, prop):
    for a, b in combinations(ALL_CODES, 2):
        assert _dice(substituted_grids, a, b, prop) == pytest.approx(
            _dice(substituted_grids, b, a, prop)
        )


def test_polarizability_overlap_is_within_unit_interval(substituted_grids):
    # all polarizability values are positive, so overlaps must be in [0, 1]
    for a, b in combinations(ALL_CODES, 2):
        assert 0.0 <= _dice(substituted_grids, a, b, "polarizability") <= 1.0


@pytest.mark.parametrize("prop", PROPS)
def test_dice_matrix_matches_snapshot(substituted_grids, prop):
    for (a, b), expected in SNAPSHOT[prop].items():
        assert _dice(substituted_grids, a, b, prop) == pytest.approx(
            expected, abs=1e-3
        ), f"{prop} {a}-{b}"
