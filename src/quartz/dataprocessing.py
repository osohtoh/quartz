from itertools import combinations
from pathlib import Path
from typing import Any
from xml import parsers

from gridData import OpenDX
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
from rdkit.Chem.rdchem import Mol
from scipy import constants
from sklearn.cluster import DBSCAN, KMeans

from quartz.data import Grid, MoleculeData

# SECTION - Property extraction functions


# NOTE - this should work with lists of MoleculeData imo
def assign_props(mol_data: MoleculeData, props: list[str]) -> None:
    """
    Assign properties to the associated RDKit molecule. Modifies the Mol object accessed by `mol_data.rdmol`.
    """
    mol: Mol = mol_data.rdmol
    # TODO - add assigning all available props
    if props == ["all"]:
        props = list(mol_data.res.keys())
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
    with open(
        temp_dir / f"{basename}.properties.out", "r", encoding="utf-8"
    ) as f:
        lines = f.readlines()
        start_idx = (
            lines.index(
                "     #   Z          covCN         q      C6AA      α(0)\n"
            )
            + 1
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
            "           -------------------------------------------------\n",
            start_idx,
        )
        for line in lines[start_idx:end_idx]:
            parts = line.split()
            props["fukui+"].append(float(parts[-3]))
            props["fukui-"].append(float(parts[-2]))
            props["fukui0"].append(float(parts[-1]))

    return props


# SECTION - Calculation functions


def calculate_overlap_arrays(arrays: list[npt.NDArray]) -> float:
    """Calculates the average overlap for a list of arrays."""
    np_grids = [g.astype(np.float64) for g in arrays]
    combs = list(combinations(np_grids, 2))

    if len(combs) == 0:
        return np.nan

    # Calculate differences pairwise to avoid huge arrays
    total_diff = 0.0
    for a, b in combs:
        total_diff += np.sum(np.abs(a - b))

    # Sum all grids
    total_sum = sum(np.sum(np.abs(g)) for g in np_grids)

    # Avoid division by zero
    if total_sum == 0 or np.isinf(total_sum):
        return np.nan
    else:
        return np.round(1 - (total_diff / (len(combs) * total_sum)), 4)


def pair_overlap_arrays(
    reference: npt.NDArray, targets: list[npt.NDArray]
) -> list[float]:
    """Calculates overlap between a reference array and a list of target arrays."""
    scores = []
    ref_grid = reference.astype(np.float64)

    for t in targets:
        np_grid = t.astype(np.float64)

        total_diff = np.sum(np.abs(np_grid - ref_grid))
        total_sum = np.sum(np.abs(np_grid)) + np.sum(np.abs(ref_grid))

        if total_sum == 0 or np.isinf(total_sum):
            scores.append(np.nan)
        else:
            scores.append(np.round(1 - (total_diff / total_sum), 4))

    return scores


def calculate_overlap(grids: list[Grid], props: list[str]) -> dict[str, float]:
    """ """
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


def pair_overlap(main_grids: dict[str, npt.NDArray], grids: list[Grid]):
    overlap: dict[str, dict[str, float]] = {}

    for prop, grid in main_grids.items():
        overlap[prop] = {}
        for g in grids:
            np_grid = g.npGrid[prop].astype(np.float64)

            total_diff = np.sum(np.abs(np_grid - grid))
            total_sum = np.sum(np.abs(np_grid)) + np.sum(np.abs(grid))

            if total_sum == 0 or np.isinf(total_sum):
                overlap[prop][g.mol_data.pdb_code] = np.nan
            else:
                overlap[prop][g.mol_data.pdb_code] = np.round(
                    1 - (total_diff / total_sum), 4
                )

    return overlap


def calculate_average(
    grids: list[Grid], props: list[str]
) -> dict[str, npt.NDArray[np.float64]]:
    prop_averages: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array(
            [grid.npGrid[key].astype(np.float64) for grid in grids]
        )

        prop_averages[key] = np.mean(np_grids, axis=0)

    return prop_averages


# TODO - make use of docking score or \DeltaG = -RTlnK as the energy for boltzmann
def calculate_boltz_avg(
    grids: list[Grid],
    props: list[str],
    energy_key: str,
    beta: float = 1.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Function calculating Boltzmann-weighted average of grid properties. Values used for energies are specified in `energy_keys` and are retrieved from `mol_data.extra` of a `MoleculeData` object.
    energy_keys: list of keys in mol_data.extra to sum for energy calculation
    """

    prop_averages: dict[str, npt.NDArray[np.float64]] = {}
    try:
        energies = np.array(
            [float(grid.mol_data.extra[energy_key]) for grid in grids]
        )
        # print("Energies: ", energies)
    except Exception as e:
        raise ValueError(
            f"Error retrieving energies using key '{energy_key}': {e}"
        )

    max_val = np.max(-beta * energies, keepdims=True)
    exp_vals = np.exp(-beta * energies - max_val)
    partition_fn = np.sum(exp_vals)
    # print("Partition function: ", partition_fn)
    prob = exp_vals / partition_fn
    # print("Probabilities: ", prob)

    for key in props:
        scaled_grids = np.array(
            [
                grid.npGrid[key].astype(np.float64) * p
                for grid, p in zip(grids, prob)
            ]
        )
        boltz_avg = np.mean(scaled_grids, axis=0)

        # print(f"Boltzmann average for {key}: ", boltz_avg)

        prop_averages[key] = boltz_avg

    return prop_averages


def calculate_pki_avg(
    grids: list[Grid], props: list[str], drop_nans=True
) -> dict[str, npt.NDArray[np.float64]]:
    prop_averages: dict[str, npt.NDArray[np.float64]] = {}

    valid_pairs = []
    for grid in grids:
        pki = grid.mol_data.pki
        if not np.isnan(pki):
            # print(pki)
            valid_pairs.append((grid, pki))
        elif not drop_nans:
            # print("appending nan")
            valid_pairs.append((grid, np.nan))

    for key in props:
        if not valid_pairs:
            continue
        pki_weighted_avg = np.mean(
            [grid.npGrid[key] * pki for grid, pki in valid_pairs], axis=0
        )
        prop_averages[key] = pki_weighted_avg
    return prop_averages


def grid_stddev(
    grid: list[Grid],
    props: list[str],
    means: dict[str, npt.NDArray[np.float64]] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    prop_stddev: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array([g.npGrid[key].astype(np.float64) for g in grid])
        if means:
            # using np.isclose because of float magic around 0.0
            prop_stddev[key] = np.nan_to_num(np.nanstd(
                np.where(np.isclose(np_grids, 0), np.nan, np_grids),
                mean=means[key],
                axis=0,
            ))
        else:
            prop_stddev[key] = np.nan_to_num(np.nanstd(
                np.where(np.isclose(np_grids, 0), np.nan, np_grids), axis=0
            ))
    return prop_stddev


def grid_abs_stddev(
    grid: list[Grid],
    props: list[str],
    means: dict[str, npt.NDArray[np.float64]] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    prop_stddev: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array([g.npGrid[key].astype(np.float64) for g in grid])
        if means:
            # using np.isclose because of float magic around 0.0
            prop_stddev[key] = np.nan_to_num(np.nanstd(
                np.where(np.isclose(np_grids, 0), np.nan, np_grids),
                mean=np.abs(means[key]),
                axis=0,
            ))
        else:
            prop_stddev[key] = np.nan_to_num(np.nanstd(
                np.where(np.isclose(np_grids, 0), np.nan, np_grids), axis=0
            ))
    return prop_stddev


def grid_variance(
    grid: list[Grid],
    props: list[str],
    means: dict[str, npt.NDArray[np.float64]] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    prop_variance: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array([g.npGrid[key].astype(np.float64) for g in grid])
        if means:
            # using np.isclose because of float magic around 0.0
            prop_variance[key] = np.nan_to_num(np.nanvar(
                np.where(np.isclose(np_grids, 0), np.nan, np_grids),
                mean=means[key],
                axis=0,
            ))
        else:
            prop_variance[key] = np.nan_to_num(np.nanvar(
                np.where(np.isclose(np_grids, 0), np.nan, np_grids), axis=0
            ))
    return prop_variance

def get_cluster_num(grid: list[Grid], prop: str, n_max: int = 10) -> int:
    wcss = []
    for k in range(1, n_max):
        kmeans = KMeans(n_clusters=k, random_state=42)
        data = np.array([g.npGrid[prop].flatten() for g in grid]).T
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Elbow method to determine optimal number of clusters
    deltas = np.diff(wcss)
    second_deltas = np.diff(deltas)
    optimal_k = np.argmax(second_deltas) + 2  # +2 to account for the diff offsets
    return int(optimal_k)

def kmeans_clustering(grid: list[Grid], props: list[str], n_clusters: int | None = None):
    clustering_results: dict[str, npt.NDArray[np.int32]] = {}

    for key in props:
        if n_clusters is None:
            n_clusters = get_cluster_num(grid, key)
        data = np.array([g.npGrid[key].flatten() for g in grid]).T
        # Seeded so clusters stay the same
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        clustering_results[key] = kmeans.labels_.astype(np.int32).reshape(
            grid[0].npGrid[key].shape
        )

    return clustering_results

def hdbscan_clustering(grid: list[Grid], props: list[str]):
    clustering_results: dict[str, Any] = {}

    for key in props:
        dbs = DBSCAN(eps=0.3)
        
    pass