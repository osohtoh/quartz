"""Baseline test for create_grid using an ethane molecule."""

import numpy as np
import pytest
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem.rdmolops import AddHs

from quartz.data import MoleculeData
from quartz.grids import create_grid


def _make_ethane_moldata() -> MoleculeData:
    """Build an ethane MoleculeData with dummy atomic properties."""
    mol = AddHs(MolFromSmiles("CC"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

    # Assign two dummy per-atom props: "charges" and "polarizibility"
    charges = [0.1, -0.1, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05]
    polar = [1.0, 1.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetProp("charges", str(charges[idx]))
        mol.GetAtomWithIdx(idx).SetProp("polarizibility", str(polar[idx]))

    return MoleculeData(
        atomic_numbers=atomic_numbers,
        coordinates=coords,
        res={},
        smiles="CC",
        rdmol=mol,
        filename="ethane",
    )


@pytest.fixture
def ethane():
    return _make_ethane_moldata()


class TestCreateGrid:
    PROPS = ["charges", "polarizibility"]

    def test_returns_one_grid_per_molecule(self, ethane):
        grids = create_grid([ethane], self.PROPS, size=1.0, padding=3.0)
        assert len(grids) == 1

    def test_grid_has_expected_properties(self, ethane):
        grids = create_grid([ethane], self.PROPS, size=1.0, padding=3.0)
        g = grids[0]
        assert set(g.npGrid.keys()) == set(self.PROPS)

    def test_grid_shape_is_3d(self, ethane):
        grids = create_grid([ethane], self.PROPS, size=1.0, padding=3.0)
        for key in self.PROPS:
            assert grids[0].npGrid[key].ndim == 3

    def test_grid_not_all_zeros(self, ethane):
        grids = create_grid([ethane], self.PROPS, size=1.0, padding=3.0)
        for key in self.PROPS:
            assert np.any(grids[0].npGrid[key] != 0), f"Grid for {key} is all zeros"

    def test_grid_values_snapshot(self, ethane):
        """Validate grid sums against baseline to detect regressions after refactor."""
        grids = create_grid(
            [ethane], self.PROPS, size=1.0, padding=3.0,
            threshold=0.01, cutoff_dist=10.0, clamp_dist=0.3, falloff_exp=12.0,
        )
        g = grids[0]
        sums = {key: float(np.sum(g.npGrid[key].astype(np.float64))) for key in self.PROPS}
        # Baseline values captured from pre-refactor run (float16 output)
        # rtol=0.01 allows for float16→float32 precision changes
        expected = {"charges": -0.0050048828125, "polarizibility": 18.31017303466797}
        for key in self.PROPS:
            np.testing.assert_allclose(sums[key], expected[key], rtol=0.01,
                                       err_msg=f"Regression in grid sum for {key}")

    def test_multiple_molecules(self, ethane):
        ethane2 = _make_ethane_moldata()
        grids = create_grid([ethane, ethane2], self.PROPS, size=1.0, padding=3.0)
        assert len(grids) == 2
        # Both grids should have same shape (same bounding box)
        for key in self.PROPS:
            assert grids[0].npGrid[key].shape == grids[1].npGrid[key].shape

    def test_grid_props_metadata(self, ethane):
        grids = create_grid([ethane], self.PROPS, size=1.0, padding=3.0)
        gp = grids[0].grid_props
        assert "origin" in gp
        assert "size" in gp
        assert gp["size"] == 1.0
