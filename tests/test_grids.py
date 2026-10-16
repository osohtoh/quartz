"""create_grid and to_dxgrid: geometry, padding, and analytic peak values."""

import numpy as np
import pytest
from gridData import Grid as DXGrid
from rdkit import Chem
from rdkit.Geometry import Point3D

from molecule_factory import PROPS, build_substituted_benzene
from quartz.data import MoleculeData
from quartz.grids import create_grid, to_dxgrid


def _single_atom(value: float, pdb_code: str = "ATM") -> MoleculeData:
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom("C"))
    rw.GetAtomWithIdx(0).SetProp("polarizability", str(value))
    mol = rw.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)
    return MoleculeData(
        atomic_numbers=np.array([6]),
        coordinates=np.zeros((1, 3)),
        res={},
        smiles="C",
        rdmol=mol,
        filename=pdb_code,
        pdb_code=pdb_code,
    )


def test_returns_one_grid_per_molecule_with_all_props():
    mol = build_substituted_benzene("chlorobenzene", "CLB")
    grids = create_grid([mol], PROPS, size=1.0, padding=3.0)
    assert len(grids) == 1
    assert set(grids[0].npGrid) == set(PROPS)
    for prop in PROPS:
        assert grids[0].npGrid[prop].ndim == 3
        assert np.any(grids[0].npGrid[prop] != 0)


def test_bounds_are_molecule_extent_plus_padding_once():
    mol = build_substituted_benzene("1,4-dichlorobenzene", "D14")
    padding = 3.0
    grid = create_grid([mol], PROPS, size=1.0, padding=padding)[0]

    lo = mol.coordinates.min(axis=0) - padding
    hi = mol.coordinates.max(axis=0) + padding
    np.testing.assert_allclose(grid.grid_props["origin"], lo)
    np.testing.assert_allclose([grid.grid_props[f"bounds_{ax}"][1] for ax in "xyz"], hi)
    expected_shape = tuple(int(np.ceil(extent)) for extent in (hi - lo))
    assert grid.npGrid["polarizability"].shape == expected_shape


def test_molecules_in_one_call_share_grid_box():
    mols = [
        build_substituted_benzene("benzene", "BEN"),
        build_substituted_benzene("1,2-dichlorobenzene", "D12"),
    ]
    grids = create_grid(mols, PROPS, size=1.0, padding=3.0)
    assert grids[0].grid_props["origin"] == grids[1].grid_props["origin"]
    for prop in PROPS:
        assert grids[0].npGrid[prop].shape == grids[1].npGrid[prop].shape


def test_grid_props_metadata():
    mol = build_substituted_benzene("benzene", "BEN")
    gp = create_grid([mol], PROPS, size=1.0, padding=3.0)[0].grid_props
    assert gp["size"] == 1.0
    assert gp["padding"] == 3.0
    assert {"origin", "bounds_x", "bounds_y", "bounds_z"} <= set(gp)


def test_single_atom_peak_equals_property_value():
    grid = create_grid([_single_atom(1.0)], ["polarizability"], size=1.0, padding=3.0)[
        0
    ]
    origin = np.array(grid.grid_props["origin"])
    atom_idx = tuple(np.round(-origin).astype(int))  # voxel sitting on (0, 0, 0)
    assert grid.npGrid["polarizability"][atom_idx] == pytest.approx(1.0)


def test_single_atom_falloff_follows_exponent_12():
    # Two voxels away the falloff is 1 / 2**12 (falloff_exp=12, clamp not active)
    grid = create_grid([_single_atom(1.0)], ["polarizability"], size=1.0, padding=3.0)[
        0
    ]
    origin = np.array(grid.grid_props["origin"])
    centre = np.round(-origin).astype(int)
    two_away = tuple(centre + np.array([2, 0, 0]))
    assert grid.npGrid["polarizability"][two_away] == pytest.approx(2.0**-12, rel=1e-3)


def test_negative_property_gives_negative_peak():
    grid = create_grid([_single_atom(-1.0)], ["polarizability"], size=1.0, padding=3.0)[
        0
    ]
    origin = np.array(grid.grid_props["origin"])
    centre = tuple(np.round(-origin).astype(int))
    assert grid.npGrid["polarizability"][centre] == pytest.approx(-1.0)


def _atom_voxel(grid) -> tuple[int, ...]:
    origin = np.array(grid.grid_props["origin"])
    return tuple(np.round(-origin).astype(int))


def _two_away(grid) -> tuple[int, ...]:
    return tuple(np.array(_atom_voxel(grid)) + np.array([2, 0, 0]))


def test_default_threshold_keeps_small_values():
    grid = create_grid([_single_atom(1.0)], ["polarizability"], size=1.0, padding=3.0)[
        0
    ]
    # 2**-12 should survive the default 0.0 threshold
    assert grid.npGrid["polarizability"][_two_away(grid)] == pytest.approx(
        2.0**-12, rel=1e-3
    )


def test_default_threshold_matches_explicit_zero():
    mol = build_substituted_benzene("chlorobenzene", "CLB")
    default = create_grid([mol], PROPS, size=1.0, padding=3.0)[0]
    explicit = create_grid([mol], PROPS, size=1.0, padding=3.0, threshold=0.0)[0]
    for prop in PROPS:
        np.testing.assert_array_equal(default.npGrid[prop], explicit.npGrid[prop])


def test_threshold_zeroes_values_below_it():
    grid = create_grid(
        [_single_atom(1.0)], ["polarizability"], size=1.0, padding=3.0, threshold=0.01
    )[0]
    values = grid.npGrid["polarizability"]
    assert values[_atom_voxel(grid)] == pytest.approx(1.0)
    assert values[_two_away(grid)] == 0.0


def test_threshold_leaves_no_nonzero_value_below_it():
    mol = build_substituted_benzene("chlorobenzene", "CLB")
    default = create_grid([mol], PROPS, size=1.0, padding=3.0)[0]
    grid = create_grid([mol], PROPS, size=1.0, padding=3.0, threshold=0.05)[0]
    for prop in PROPS:
        values = grid.npGrid[prop]
        assert np.all((values == 0) | (np.abs(values) >= 0.05))
        assert np.count_nonzero(values) < np.count_nonzero(default.npGrid[prop])


def test_threshold_uses_absolute_value():
    grid = create_grid(
        [_single_atom(-1.0)], ["polarizability"], size=1.0, padding=3.0, threshold=0.01
    )[0]
    values = grid.npGrid["polarizability"]
    assert values[_atom_voxel(grid)] == pytest.approx(-1.0)
    assert values[_two_away(grid)] == 0.0


def test_threshold_keeps_values_equal_to_it():
    default = create_grid(
        [_single_atom(1.0)], ["polarizability"], size=1.0, padding=3.0
    )[0]
    value = float(default.npGrid["polarizability"][_two_away(default)])

    kept = create_grid(
        [_single_atom(1.0)], ["polarizability"], size=1.0, padding=3.0, threshold=value
    )[0]
    dropped = create_grid(
        [_single_atom(1.0)],
        ["polarizability"],
        size=1.0,
        padding=3.0,
        threshold=value * 1.01,
    )[0]
    assert kept.npGrid["polarizability"][_two_away(kept)] == pytest.approx(value)
    assert dropped.npGrid["polarizability"][_two_away(dropped)] == 0.0


def test_scalar_threshold_applies_to_every_property():
    mol = build_substituted_benzene("chlorobenzene", "CLB")
    default = create_grid([mol], PROPS, size=1.0, padding=3.0)[0]
    grid = create_grid([mol], PROPS, size=1.0, padding=3.0, threshold=0.5)[0]
    for prop in PROPS:
        base = default.npGrid[prop]
        np.testing.assert_array_equal(
            grid.npGrid[prop], np.where(np.abs(base) < 0.5, 0.0, base)
        )


def test_dict_threshold_applies_per_property_and_defaults_missing_to_zero():
    mol = build_substituted_benzene("chlorobenzene", "CLB")
    default = create_grid([mol], PROPS, size=1.0, padding=3.0)[0]
    grid = create_grid(
        [mol], PROPS, size=1.0, padding=3.0, threshold={"polarizability": 0.5}
    )[0]
    np.testing.assert_array_equal(grid.npGrid["charges"], default.npGrid["charges"])
    base = default.npGrid["polarizability"]
    np.testing.assert_array_equal(
        grid.npGrid["polarizability"], np.where(np.abs(base) < 0.5, 0.0, base)
    )


def test_threshold_dict_is_not_modified():
    thresholds = {"polarizability": 0.5}
    create_grid([_single_atom(1.0)], ["polarizability"], threshold=thresholds)
    assert thresholds == {"polarizability": 0.5}


@pytest.mark.parametrize("threshold", [-0.1, {"polarizability": -0.1}, "0.1"])
def test_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError):
        create_grid([_single_atom(1.0)], ["polarizability"], threshold=threshold)


def test_to_dxgrid_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    to_dxgrid(arr, origin=(1.0, 2.0, 3.0), spacing=0.5, filename="mask.dx")

    dx = DXGrid(tmp_path / "grids" / "data_mask.dx")
    np.testing.assert_allclose(dx.grid, arr)
    np.testing.assert_allclose(dx.origin, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(dx.delta, [0.5, 0.5, 0.5])
