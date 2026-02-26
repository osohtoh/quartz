from dataclasses import dataclass, field
from typing import Any
import numpy.typing as npt
import numpy as np
import qcelemental.models.types as qcelt
from gridData import OpenDX
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D, UniformGrid3D_


@dataclass
class MoleculeData:
    atomic_numbers: qcelt.Array
    coordinates: qcelt.Array
    res: dict[str, Any] # xtb legacy thingy
    smiles: str
    rdmol: Mol
    filename: str = ""
    pdb_code: str = ""
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
        if self.mol_data and self.mol_data.extra and "pki" in self.mol_data.extra:
            pki_val = self.mol_data.extra["pki"]
            if pki_val is not None:
                return pki_val
        return np.nan

    @property
    def ki(self) -> float:
        if self.mol_data and self.mol_data.extra and "ki" in self.mol_data.extra:
            ki_val = self.mol_data.extra["ki"]
            if ki_val is not None:
                return ki_val
        return np.nan

    @property
    def weight(self) -> float:
        if self.mol_data and self.mol_data.extra and "weight" in self.mol_data.extra:
            weight_val = self.mol_data.extra["weight"]
            if weight_val is not None:
                return weight_val
        return np.nan

    @property
    def extra(self) -> dict[str, Any] | None:
        if self.mol_data and self.mol_data.extra:
            return self.mol_data.extra
        return None


def pki_grid_scale(grid: Grid) -> Grid | None:
    if grid.pki is None or np.isnan(grid.pki):
        return None
    scaled_grids = {}
    grid_ret = grid
    for key, prop_grid in grid.npGrid.items():
        scaled_grids[key] = prop_grid * grid.pki
    grid_ret.npGrid = scaled_grids
    return grid_ret


def set_weights(grids: list[Grid], weight_key: str) -> None:
    values = [grid.extra[weight_key] for grid in grids]
    # print("values: ", values)
    weights = np.array(values) / np.nansum(values)
    # print("weights: ", weights)
    for i, grid in enumerate(grids):
        if (
            grid.mol_data is not None
            and grid.mol_data.extra is not None
            and "weight" not in grid.mol_data.extra
        ):
            # print("assigning weight: ", weights[i])
            grid.mol_data.extra["weight"] = weights[i]

def weight_scale(grid: Grid) -> Grid:
    if grid.weight is None or np.isnan(grid.weight):
        return grid
    scaled_grids = {}
    for key, prop_grid in grid.npGrid.items():
        scaled_grids[key] = prop_grid * grid.weight
    grid.npGrid = scaled_grids
    return grid