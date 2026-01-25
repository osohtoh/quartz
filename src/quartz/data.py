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
    ki: float | None = None
    pki: float | None = None
    extra: dict[str, Any] | None = None


@dataclass
class Grid:
    npGrid: dict[str, npt.NDArray] = field(repr=False)
    grid_props: dict[str, Any]
    rdGrid: UniformGrid3D_ | None = field(default=None, repr=False)
    prop_map: list[tuple[int, str]] | None = None
    points: list[Point3D] | None = None
    point_idxs: list[tuple[int]] | None = None
    mol_data: MoleculeData | None = None
    dxGrid: OpenDX.field | None = None

    @property
    def pki(self) -> float:
        if self.mol_data and self.mol_data.pki is not None:
            return self.mol_data.pki
        return float("nan")

    @property
    def ki(self) -> float:
        if self.mol_data and self.mol_data.ki is not None:
            return self.mol_data.ki
        return float("nan")
