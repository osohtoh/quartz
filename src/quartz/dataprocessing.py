from itertools import combinations

import numpy as np
from rdkit.Chem.rdchem import Mol

from quartz.io import Grid, MoleculeData


# NOTE - this should work with lists of MoleculeData imo
def assign_props(mol_data: MoleculeData, props: list[str]) -> None:
    """
    Assign properties to the associated RDKit molecule. Modifies the Mol object accessed by `mol_data.rdmol`.
    """
    mol: Mol = mol_data.rdmol
    # TODO - add assigning all available props
    for key in props:
        # TODO - check whether key matches tblite/dftd4 docs
        for idx, val in enumerate(mol_data.res[key]):
            mol.GetAtomWithIdx(idx).SetProp(key, str(val))
    mol_data.rdmol = mol


def calculate_overlap(grids: list[Grid], props: list[str]) -> dict[str, float]:
    prop_overlap: dict[str, float] = {}

    for key in props:
        np_grids = [grid.npGrid[key] for grid in grids]
        combs = combinations(np_grids, 2)
        differences = np.array([np.subtract(a, b) for a, b in combs])
        sums = np.sum(np.array(np_grids))
        prop_overlap[key] = np.round(
            1 - np.abs(np.sum(differences) / (len(differences) * sums)), 4
        )

    return prop_overlap
