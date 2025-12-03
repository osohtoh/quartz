import shutil
import subprocess
from pathlib import Path

from rdkit.Chem.Descriptors import NumRadicalElectrons
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem.rdmolfiles import MolFromPDBFile

from quartz.dataprocessing import extract_props
from quartz.utils import xyz_to_mol


def generate_input(input_path: Path, size: float, charge: int, uhf: int):
    input_file = Path(input_path) / "input.inp"
    if not input_file.exists():
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
        input_file.write_text(input)


def get_props(input_geo: str, size: float = 1.0):
    """ """
    # Create temp directory
    input_path = Path(input_geo)
    temp_dir = input_path.parent / f"temp_{input_path.stem}"
    temp_dir = temp_dir.resolve()  # absolute path

    try:
        temp_dir.mkdir(exist_ok=True)  # More robust than os.mkdir()
    except PermissionError as e:
        raise PermissionError(f"Cannot create {temp_dir}: {e}")

    # Load molecule
    try:
        rdmol = MolFromPDBFile(input_geo, removeHs=False)
    except Exception:
        rdmol = xyz_to_mol(input_geo)

    # Determine charge and unpaired electrons
    charge = GetFormalCharge(rdmol)
    uhf = NumRadicalElectrons(rdmol)

    # Create input file if it doesn't exist
    generate_input(temp_dir, size, charge, uhf)

    file = input_path.name

    relative_parent = Path("..") / file

    # Run XTB
    result = subprocess.run(
        [
            "xtb",
            "-I",
            "input.inp",
            "--molden",
            "--namespace",
            Path(input_geo).stem,
            str(relative_parent),
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
            str(relative_parent),
        ],
        capture_output=True,
        cwd=temp_dir,
    )

    result_fod = subprocess.run(
        [
            "xtb",
            "--fod",
            "--etemp",
            "5000",  # as per docs, might need adjusting
            "--namespace",
            Path(input_geo).stem,
            str(relative_parent),
        ],
        capture_output=True,
        cwd=temp_dir,
    )

    (temp_dir / f"{input_path.stem}.out").write_bytes(result.stdout)
    (temp_dir / f"{input_path.stem}_fukui.out").write_bytes(result_fuk.stdout)
    (temp_dir / f"{input_path.stem}_fod.out").write_bytes(result_fod.stdout)

    if result.returncode or result_fuk.returncode or result_fod.returncode:
        print(result.stdout)
        raise Exception("xTB calculation failed.")
        
    # Parse output and remove temp dir
    out = extract_props(temp_dir, input_path.stem)
    # shutil.rmtree(temp_dir)

    return out
