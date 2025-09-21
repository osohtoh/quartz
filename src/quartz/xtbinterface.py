import os
import subprocess
from pathlib import Path

from rdkit.Chem.Descriptors import NumRadicalElectrons
from rdkit.Chem.rdmolops import GetFormalCharge

from quartz.dataprocessing import extract_props
from quartz.utils import xyz_to_mol


def generate_input(input_path: str, size: float, charge: int, uhf: int):
    if not os.path.exists(os.path.join(input_path, "input.inp")):
        input = f"""$chrg {charge}
$spin {uhf}
$gfn
   method=2
   scc=true
   periodic=false
   dispscale=1.000000000000000
$scc
   maxiterations=250
   temp=300
   broydamp=.4
$cube
   step={size}
   cal=1
$write
   density=true
   spin population=true
   spin density=true
   charges=true
   mulliken=true
   orbital energies=true
   wiberg=true
   output file=properties.out
   json=true
   esp=true
$end
        """
        with open(os.path.join(input_path, "input.inp"), "w") as f:
            f.write(input)


# NOTE maybe change to pathlib fully, low pri
def get_props(input_geo: str, size: float = 1.0):
    """ """
    # Create temp directory
    temp_dir = os.path.abspath(
        os.path.join(os.path.dirname(input_geo), f"temp_{Path(input_geo).stem}")
    )
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    # Load molecule
    rdmol = xyz_to_mol(input_geo)

    # Determine charge and unpaired electrons
    charge = GetFormalCharge(rdmol)
    uhf = NumRadicalElectrons(rdmol)

    # Create input file if it doesn't exist
    generate_input(temp_dir, size, charge, uhf)

    basename = os.path.basename(input_geo)

    # Run XTB
    result = subprocess.run(
        [
            "xtb",
            "-I",
            "input.inp",
            "--molden",
            "--namespace",
            Path(input_geo).stem,
            os.path.relpath(f"../{basename}"),
        ],
        capture_output=True,
        cwd=temp_dir,
    )

    result_fuk = subprocess.run(
        [
            "xtb",
            "--vfukui",
            "--namespace",
            Path(input_geo).stem,
            os.path.relpath(f"../{basename}"),
        ],
        capture_output=True,
        cwd=temp_dir,
    )

    result_fod = subprocess.run(
        [
            "xtb",
            "--fod",
            "--etemp",
            "5000", # as per docs, might need adjusting
            "--namespace",
            Path(input_geo).stem,
            os.path.relpath(f"../{basename}"),
        ],
        capture_output=True,
        cwd=temp_dir,
    )

    with open(os.path.join(temp_dir, f"{Path(input_geo).stem}.out"), "wb") as f:
        f.write(result.stdout)

    with open(os.path.join(temp_dir, f"{Path(input_geo).stem}_fukui.out"), "wb") as f:
        f.write(result_fuk.stdout)

    with open(os.path.join(temp_dir, f"{Path(input_geo).stem}_fod.out"), "wb") as f:
        f.write(result_fod.stdout)

    if result.returncode or result_fuk.returncode or result_fod.returncode:
        print(result.stdout)
        result.check_returncode()
    # Parse output

    extract_props(basename)

    return {}
