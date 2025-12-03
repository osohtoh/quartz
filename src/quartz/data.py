from dataclasses import dataclass, field
from typing import Any
import numpy.typing as npt
import qcelemental.models.types as qcelt
from gridData import OpenDX
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D, UniformGrid3D_

@dataclass
class MoleculeData:
    atomic_numbers: qcelt.Array
    coordinates: qcelt.Array
    res: dict[str, Any]
    smiles: str
    rdmol: Mol
    filename: str = ""
    pdb_code: str = ""
    ki: float = 0.0
    pki: float = 0.0


@dataclass
class Grid:
    npGrid: dict[str, npt.NDArray]
    rdGrid: UniformGrid3D_
    prop_map: list[tuple[int, str]]
    points: list[Point3D]
    point_idxs: list[tuple[int]]
    grid_props: list[list[float]]
    mol_data: MoleculeData
    dxGrid: OpenDX.field | None = None
