from dataclasses import dataclass, field
from typing import Any
import numpy.typing as npt
import numpy as np
import qcelemental.models.types as qcelt
from gridData import OpenDX
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D, UniformGrid3D_
import scipy.constants


# can I do data validation with pydantic
@dataclass
class MoleculeData:
    atomic_numbers: qcelt.Array
    coordinates: qcelt.Array
    res: dict[str, Any]  # xtb legacy thingy
    smiles: str  # probably better in extra
    rdmol: Mol
    filename: str = ""  # pathlib Path
    pdb_code: str = ""  # should just be named id
    extra: dict[str, Any] | None = None


# TODO - possible big thing, dict of ID: Grid
# Serialize to HDF5 (probably), ID Code -> Prop -> Array
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
    def pki(self):
        if self.mol_data and self.mol_data.extra and "pki" in self.mol_data.extra:
            pki_val = self.mol_data.extra["pki"]
            if pki_val is not None:
                return pki_val
        return np.nan

    @property
    def ki(self):
        if self.mol_data and self.mol_data.extra and "ki" in self.mol_data.extra:
            ki_val = self.mol_data.extra["ki"]
            if ki_val is not None:
                return ki_val
        return np.nan

    @property
    def deltaG(self):
        if (
            self.mol_data
            and self.mol_data.extra
            and "deltaG" in self.mol_data.extra
            and self.mol_data.extra["deltaG"] is not None
        ):
            return self.mol_data.extra["deltaG"]
        if (
            self.mol_data
            and self.mol_data.extra
            and "ki" in self.mol_data.extra
            and "temp" in self.mol_data.extra
        ):
            ki = self.mol_data.extra["ki"]
            temp = self.mol_data.extra["temp"]
            if ki is not None and temp is not None:
                return -scipy.constants.R * temp * np.log(ki)

    @property
    def weight(self):
        if self.mol_data and self.mol_data.extra and "weight" in self.mol_data.extra:
            weight_val = self.mol_data.extra["weight"]
            if weight_val is not None:
                return weight_val
        return np.nan

    @weight.setter
    def weight(self, val: float):
        self._weight = val

    @property
    def extra(self) -> dict[str, Any] | None:
        if self.mol_data and self.mol_data.extra:
            return self.mol_data.extra
        return None

    @property
    def id_code(self) -> str:
        if self.mol_data and self.mol_data.pdb_code:
            return self.mol_data.pdb_code
        elif self.mol_data and self.mol_data.filename:
            return self.mol_data.filename
        else:
            return "unknown_id"

    def set_weight(self, weight_key: str):
        if self.mol_data and self.mol_data.extra is None:
            raise Exception("No 'extra' dict found")
        if self.extra and weight_key in self.extra:
            self.weight = self.extra[weight_key]

    def weigh_grid(self):
        if self.extra and self.extra["weight"] is not None:
            self.npGrid = {
                prop: self.npGrid[prop] * self.extra["weight"]
                for prop in self.npGrid.keys()
            }


# Scales pKi and grids between 0 and 1, and applies the weight
def scale_grids(grids: list[Grid]):
    """
    Takes a list of grids, scales pKi between 0 and 1
    """
    pkis = [grid.pki for grid in grids]
    max_pki = max(pkis)
    for grid in grids:
        if grid.extra is None:
            print(f"Skipping grid {grid.id_code}. No 'extra' dict found.")
            continue
        grid.extra["weight"] = grid.pki / max_pki
        grid.weigh_grid()
    return grids


def pki_grid_scale(grid: Grid) -> Grid | None:
    if grid.pki is None or np.isnan(grid.pki):
        return None
    scaled_grids = {}
    for key, prop_grid in grid.npGrid.items():
        scaled_grids[key] = prop_grid * grid.pki
    return Grid(
        npGrid=scaled_grids,
        grid_props=grid.grid_props,
        rdGrid=grid.rdGrid,
        prop_map=grid.prop_map,
        points=grid.points,
        point_idxs=grid.point_idxs,
        mol_data=grid.mol_data,
        dxGrid=grid.dxGrid,
    )


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
