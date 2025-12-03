from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit.Chem.rdchem import Mol

from quartz.data import Grid, MoleculeData


# NOTE - this should work with lists of MoleculeData imo
def assign_props(mol_data: MoleculeData, props: list[str]) -> None:
    """
    Assign properties to the associated RDKit molecule. Modifies the Mol object accessed by `mol_data.rdmol`.
    """
    mol: Mol = mol_data.rdmol
    # TODO - add assigning all available props
    for key in props:
        # TODO - check whether key matches tblite/dftd4 docs
        for idx, val in enumerate(mol_data.res[key]):
            mol.GetAtomWithIdx(idx).SetProp(key, str(val))
    mol_data.rdmol = mol


def extract_props(temp_dir: Path, basename: str):
    # .out files have properties per atom
    # From main basename.properties.out:
    # - coordination number covCN (?)
    # - dispersion coeff C6
    # - polarizability α
    # [     #   Z          covCN         q      C6AA      α(0)]
    # - Wiberg bond orders (?), also from .wbo file
    # From basename_fod.out:
    # - fractional occupation density per atom per shell
    # [ Loewdin FODpop      n(s)   n(p)   n(d)]
    # From basename_fukui.out:
    # - Fukui functions per atom (f+, f-, f0)
    # [     #        f(+)     f(-)     f(0)]
    # From basename.charges:
    # - partial charge q
    
    props = {
        "coordination number": [],
        "dispersion coef": [],
        "polarizibility": [],
        "charges": [],
        "fullpop": [],
        "fod_s": [],
        "fod_p": [],
        "fod_d": [],
        "fukui+": [],
        "fukui-": [],
        "fukui0": [],
    }

    # print(f"Extracting properties for {basename}...")

    # FIXME - change so it uses the json file

    # Process .out file
    with open(temp_dir / f"{basename}.properties.out", "r", encoding="utf-8") as f:
        lines = f.readlines()
        start_idx = (
            lines.index("     #   Z          covCN         q      C6AA      α(0)\n") + 1
        )
        end_idx = lines.index("\n", start_idx)
        for line in lines[start_idx:end_idx]:
            parts = line.split()
            props["coordination number"].append(float(parts[3]))
            props["dispersion coef"].append(float(parts[5]))
            props["polarizibility"].append(float(parts[6]))

    # Process .charges file
    with open(temp_dir / f"{basename}.charges", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            props["charges"].append(float(line))

    # Process _fod.out file
    with open(temp_dir / f"{basename}_fod.out", "r", encoding="utf-8") as f:
        lines = f.readlines()
        start_idx = lines.index(" Loewdin FODpop      n(s)   n(p)   n(d)\n") + 1
        end_idx = lines.index("\n", start_idx)
        for line in lines[start_idx:end_idx]:
            parts = line.split()
            props["fullpop"].append(float(parts[2]))
            # You could do falloffs specific to type of orbital (probably not needed, but cool idea nonetheless)
            props["fod_s"].append(float(parts[-3]))
            props["fod_p"].append(float(parts[-2]))
            props["fod_d"].append(float(parts[-1]))

    # Process _fukui.out file
    with open(temp_dir / f"{basename}_fukui.out", "r", encoding="utf-8") as f:
        lines = f.readlines()
        start_idx = lines.index("     #        f(+)     f(-)     f(0)\n") + 1
        end_idx = lines.index(
            "           -------------------------------------------------\n", start_idx
        )
        for line in lines[start_idx:end_idx]:
            parts = line.split()
            props["fukui+"].append(float(parts[-3]))
            props["fukui-"].append(float(parts[-2]))
            props["fukui0"].append(float(parts[-1]))


    return props


def calculate_overlap(grids: list[Grid], props: list[str]) -> dict[str, float]:
    prop_overlap: dict[str, float] = {}

    for key in props:
        # Convert to float64 to prevent overflow
        np_grids = [grid.npGrid[key].astype(np.float64) for grid in grids]
        combs = list(combinations(np_grids, 2))
        
        if len(combs) == 0:
            prop_overlap[key] = np.nan
            continue
        
        # Calculate differences pairwise to avoid huge arrays
        total_diff = 0.0
        for a, b in combs:
            total_diff += np.sum(np.abs(a - b))
        
        # Sum all grids
        total_sum = sum(np.sum(np.abs(g)) for g in np_grids)
        
        # Avoid division by zero
        if total_sum == 0 or np.isinf(total_sum):
            prop_overlap[key] = np.nan
        else:
            prop_overlap[key] = np.round(
                1 - (total_diff / (len(combs) * total_sum)), 4
            )

    return prop_overlap
