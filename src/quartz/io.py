import glob
from quartz.xtbinterface import get_props
from quartz.utils import xyz_to_mol, get_smiles, get_mol, optimize
from quartz.data import MoleculeData
import os

from qcelemental.models.molecule import Molecule
import qcelemental as qcel


def load_and_process(
    smiles: list[str] | None = None, input_dir: str | None = None, size: float = 1.0
) -> list[MoleculeData]:
    """
    Load and process molecules from either a list of SMILES strings or a directory of
    xyz and/or pdb files. The files are expected to have hydrogens and be optimized.
    For charged molecules it is recommended to use pdb files.

    Args:
        smiles: List of SMILES strings
        input_dir: Path to directory of xyz/pdb files

    Returns:
        list[MoleculeData]: A list of MoleculeData objects containing the loaded
            molecules with their properties calculated by the xTB interface.

    Raises:
        Exception: If neither a list of SMILES strings nor a directory path is provided
    """
    output: list[MoleculeData] = []

    if not smiles and not input_dir:
        raise Exception

    # Path input expected to have hydrogens and be optimized
    if input_dir:
        paths = glob.glob(f"{input_dir}/*.*")
        for path in paths:
            mol: Molecule = get_mol(path)
            res = get_props(path, size)
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