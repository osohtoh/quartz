import time
from pathlib import Path
from typing import Union

import psi4
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons


def setup_psi4_environment(memory: str = "32GB", threads: int = 16):
    """
    Sets up the global Psi4 environment, including resource limits
    and halogen bond distance overrides to fix Psi4 bond perception.
    """
    psi4.set_memory(memory)
    psi4.set_num_threads(threads)

    # Override Psi4 default bond perceptions for halogens
    psi4.qcdb.parker._expected_bonds["CL"] = 1
    psi4.qcdb.parker._expected_bonds["BR"] = 1
    psi4.qcdb.parker._expected_bonds["I"] = 1


def prepare_psi4_geometry(filepath: Union[str, Path]) -> str:
    """
    Loads a .mol, .pdb, or .xyz file using RDKit and converts it to a
    Psi4 input geometry string. Raises ValueError for a molecule with multiple fragments.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    # Load molecule based on extension
    if ext == ".mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
    elif ext == ".xyz":
        # RDKit's ForwardSDMolSupplier can read xyz, or use MolFromXYZFile
        mol = Chem.MolFromXYZFile(str(path))
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Only .mol, .pdb, and .xyz are supported."
        )

    if mol is None:
        raise ValueError(f"Failed to parse molecule from {path}")

    # Validate that we have exactly one fragment
    # frags = Chem.GetMolFrags(mol)
    # if len(frags) > 1:
    #     raise ValueError(
    #         f"Molecule contains {len(frags)} fragments. Only single fragments are supported."
    #     )

    # Calculate charge and multiplicity
    charge = Chem.GetFormalCharge(mol)
    multiplicity = NumRadicalElectrons(mol) + 1

    # Build geometry string
    geom_lines = [f"{charge} {multiplicity}"]
    conf = mol.GetConformer()
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        geom_lines.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

    # Standardize symmetry
    geom_lines.append("symmetry c1")

    return "\n".join(geom_lines)


def calculate_wavefunction(
    geometry_str: str,
    functional: str,
    basis: str,
    output_wfn_path: Union[str, Path],
    n_retries: int = 3,
):
    """
    Executes a Psi4 energy calculation from a geometry string and saves the
    Wavefunction object to a numpy file. Cleans memory and retries upon failure.
    """
    output_path = Path(output_wfn_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, n_retries + 1):
        try:
            mol = psi4.geometry(geometry_str)

            psi4.set_options({"maxiter": 999})

            energy, wfn = psi4.energy(
                f"{functional}/{basis}", molecule=mol, return_wfn=True
            )

            wfn.to_file(str(output_path))
            return output_path

        except Exception as e:
            if attempt < n_retries:
                print(
                    f"Attempt {attempt}/{n_retries} failed for wavefunction calculation. Error: {e}"
                )
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Calculation failed after {n_retries} attempts. Final error: {e}"
                ) from e
        finally:
            psi4.core.clean()
            psi4.core.clean_timers()
            psi4.core.clean_options()
