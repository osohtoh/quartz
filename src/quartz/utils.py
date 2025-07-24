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
    # NOTE - This is a bit of a hack, also needs proper charge determination
    if file.find("/") != -1:
        if file[-3:] == "xyz":
            mol = MolFromXYZFile(file)
            DetermineBonds(mol, charge=chrg)
        elif file[-3:] == "pdb":
            mol = MolFromPDBFile(file, removeHs=False)
    else:
        mol = MolFromXYZBlock(file)
        DetermineBonds(mol, charge=chrg)
    if mol is None:
        raise Exception
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
            mol = qcel.models.Molecule.from_data(pdb_to_xyz(path))
            return mol
