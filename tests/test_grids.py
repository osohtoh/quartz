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


def test_to_dxgrid_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    to_dxgrid(arr, origin=(1.0, 2.0, 3.0), spacing=0.5, filename="mask.dx")

    dx = DXGrid(tmp_path / "grids" / "data_mask.dx")
    np.testing.assert_allclose(dx.grid, arr)
    np.testing.assert_allclose(dx.origin, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(dx.delta, [0.5, 0.5, 0.5])
