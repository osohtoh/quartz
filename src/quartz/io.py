from dataclasses import dataclass
import glob
from typing import Any
from quartz.xtbinterface import get_props
from quartz.utils import xyz_to_mol, get_smiles, get_mol, optimize
import os

import numpy as npt
import numpy.typing as npt
import qcelemental.models.types as qcelt
from qcelemental.models.molecule import Molecule
import qcelemental as qcel
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
    filename: str | None


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

def load_and_process(
    smiles: list[str] | None = None, input_dir: str | None = None
) -> list[MoleculeData]:
    output: list[MoleculeData] = []

    if not smiles and not input_dir:
        raise Exception

    # Path input expected to have hydrogens and be optimized
    if input_dir:
        paths = glob.glob(f"{input_dir}/*.*")
        for path in paths:
            mol: Molecule = get_mol(path)
            res = get_props(mol)
            output.append(
                MoleculeData(
                    mol.atomic_numbers,
                    mol.geometry,
                    res,
                    get_smiles(path),
                    xyz_to_mol(path),
                    os.path.basename(path),
                )
            )
        return output

    if smiles:
        for smile in smiles:
            xyz = optimize(smile)
            mol: Molecule = qcel.models.Molecule.from_data(xyz)
            res = get_props(mol)
            output.append(
                MoleculeData(
                    mol.atomic_numbers, mol.geometry, res, smile, xyz_to_mol(xyz), None
                )
            )
        return output

    raise Exception("Failed to load the data. Have you passed a path or SMILES?")