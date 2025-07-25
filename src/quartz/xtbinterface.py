import subprocess
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem.Descriptors import NumRadicalElectrons
from quartz.utils import xyz_to_mol
import os


def generate_input(input_path: str, size: float):
    if not os.path.exists(os.path.join(input_path, "input.inp")):
        input = f"""
            $cube
                step={size}
                cal=1
            $end
            $write
                density=true
                spin density=true
                charges=true
                mulliken=true
                orbital energies=true
                wiberg=true
            $end
        """
        with open(os.path.join(input_path, "input.inp"), "w") as f:
            f.write(input)


def get_props(input_path: str, size: float):
    # Create input file if it doesn't exist
    generate_input(input_path, size)
    input_file = os.path.join(input_path, "input.inp")

    # Create temp directory
    temp_dir = os.path.join(
        os.path.dirname(input_path), f"temp_{os.path.basename(input_path)}"
    )
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Load molecule
    rdmol = xyz_to_mol(input_path)
    # Determine charge
    charge = GetFormalCharge(rdmol)
    uhf = NumRadicalElectrons(rdmol)
    # Run XTB
    result = subprocess.run(
        [f"xtb -scc --input {input_file} -c {charge} -u {uhf} --fod {input_path}"],
        capture_output=True,
        cwd=temp_dir,
    )
    result.check_returncode()
    # Parse output

    return {}
