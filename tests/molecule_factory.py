"""Deterministic substituted-benzene builders for overlap tests.

Geometry is generated directly (regular hexagon in the xy-plane, substituents
radiating outwards), so every molecule shares one frame and no embedding or
alignment step is involved. Per-atom properties are set by hand.
"""

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from quartz.data import Grid, MoleculeData

RING_RADIUS = 1.39
BOND_LENGTH = {"H": 1.08, "Cl": 1.73}
ATOMIC_NUMBER = {"H": 1, "C": 6, "Cl": 17}
PROP_VALUES = {
    "polarizability": {"H": 0.4, "C": 1.2, "Cl": 2.2},
    "charges": {"H": 0.1, "C": -0.1, "Cl": -0.2},
}
PROPS = list(PROP_VALUES)

# ring position index -> substituent; unlisted positions carry H
PATTERNS = {
    "benzene": ({}, "c1ccccc1"),
    "chlorobenzene": ({0: "Cl"}, "Clc1ccccc1"),
    "1,2-dichlorobenzene": ({0: "Cl", 1: "Cl"}, "Clc1ccccc1Cl"),
    "1,3-dichlorobenzene": ({0: "Cl", 2: "Cl"}, "Clc1cccc(Cl)c1"),
    "1,4-dichlorobenzene": ({0: "Cl", 3: "Cl"}, "Clc1ccc(Cl)cc1"),
}


def build_substituted_benzene(
    name: str,
    pdb_code: str,
    pki: float | None = None,
    doi: str = "10.0000/test",
) -> MoleculeData:
    substituents, smiles = PATTERNS[name]

    elements = ["C"] * 6
    coords = [
        np.array(
            [
                RING_RADIUS * np.cos(np.pi / 3 * i),
                RING_RADIUS * np.sin(np.pi / 3 * i),
                0.0,
            ]
        )
        for i in range(6)
    ]
    bonds = [(i, (i + 1) % 6) for i in range(6)]

    for i in range(6):
        sub = substituents.get(i, "H")
        direction = np.array([np.cos(np.pi / 3 * i), np.sin(np.pi / 3 * i), 0.0])
        bonds.append((i, len(elements)))
        elements.append(sub)
        coords.append(coords[i] + BOND_LENGTH[sub] * direction)

    rw = Chem.RWMol()
    for el in elements:
        rw.AddAtom(Chem.Atom(el))
    for a, b in bonds:
        rw.AddBond(a, b, Chem.BondType.SINGLE)
    for idx, el in enumerate(elements):
        atom = rw.GetAtomWithIdx(idx)
        for key, values in PROP_VALUES.items():
            atom.SetProp(key, str(values[el]))

    mol = rw.GetMol()
    conf = Chem.Conformer(len(elements))
    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(*xyz))
    mol.AddConformer(conf, assignId=True)

    return MoleculeData(
        atomic_numbers=np.array([ATOMIC_NUMBER[el] for el in elements]),
        coordinates=np.array(coords),
        res={},
        smiles=smiles,
        rdmol=mol,
        filename=name,
        pdb_code=pdb_code,
        extra={"pki": pki, "doi": doi},
    )


def make_dummy_grid(
    values: np.ndarray,
    pdb_code: str,
    pki: float | None = None,
    prop: str = "polarizability",
    size: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Grid:
    mol = MoleculeData(
        atomic_numbers=np.array([6]),
        coordinates=np.zeros((1, 3)),
        res={},
        smiles="C",
        rdmol=None,
        filename=pdb_code,
        pdb_code=pdb_code,
        extra={"pki": pki, "doi": "10.0000/synthetic"},
    )
    return Grid(
        npGrid={prop: np.asarray(values, dtype=np.float32)},
        grid_props={"origin": origin, "size": size},
        mol_data=mol,
    )
