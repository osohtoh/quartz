from pathlib import Path

import qcelemental as qcel
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDetermineBonds import DetermineBonds
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolfiles import (
    MolFromPDBFile,
    MolFromSmiles,
    MolFromXYZBlock,
    MolFromXYZFile,
    MolToSmiles,
    MolToXYZBlock,
)
from rdkit.Chem.rdmolops import AddHs


def optimize(smiles: str) -> str:
    mol = AddHs(MolFromSmiles(smiles))
    EmbedMolecule(mol)
    MMFFOptimizeMolecule(mol)
    xyz = MolToXYZBlock(mol)
    return xyz


def xyz_to_mol(file: str, chrg: int = 0) -> Mol:
    mol = None
    try:
        p = Path(file)
        exists = p.exists()
        suffix = p.suffix.lower()
    except Exception:
        exists = False
        suffix = ""

    # NOTE - This is a bit of a hack, also needs proper charge determination
    if "/" in file or "\\" in file or exists:
        if suffix == ".xyz" or file[-3:].lower() == "xyz":
            mol = MolFromXYZFile(file)
            DetermineBonds(mol, charge=chrg)
        elif suffix == ".pdb" or file[-3:].lower() == "pdb":
            mol = MolFromPDBFile(file, removeHs=False)
        else:
            raise Exception("Invalid file type. Use xyz file/block or pdb.")
    else:
        mol = MolFromXYZBlock(file)
        DetermineBonds(mol, charge=chrg)
    if mol is None:
        raise Exception("Failed to parse molecule from input")
    return mol


def get_smiles(path: str) -> str:
    match path[-3:]:
        case "xyz":
            return MolToSmiles(xyz_to_mol(path))
        case "pdb":
            # NOTE - we don't need Hs in the SMILES
            return MolToSmiles(MolFromPDBFile(path))
        case _:
            raise Exception("Invalid file type. Use xyz or pdb.")


def pdb_to_xyz(file: str) -> str:
    return MolToXYZBlock(MolFromPDBFile(file, removeHs=False))


def get_mol(path: str) -> Mol:
    match path[-3:]:
        case "xyz":
            mol = qcel.models.Molecule.from_file(path)
            return mol
        case "pdb":
            mol = qcel.models.Molecule.from_data(pdb_to_xyz(path), dtype="xyz")
            return mol
        case _:
            raise Exception(
                f"Unsupported file format. Choose pdb or xyz file. \n File: {path}"
            )


def get_pdb_charge(path: str) -> int:
    pass

def psiwfn_to_fchk(wfn_path):
    try:
        import psi4
    except ImportError:
        raise ImportError("Psi4 is not installed. Cannot convert wfn to fchk.")
    wfn = psi4.core.Wavefunction.from_file(wfn_path)
    psi4.driver.fchk(wfn, str(wfn_path).replace('.npy', '.fchk'))