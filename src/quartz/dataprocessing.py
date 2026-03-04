from itertools import combinations
from pathlib import Path
from typing import Any, Literal
from xml import parsers

from gridData import OpenDX
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
from rdkit.Chem.rdchem import Mol
from scipy import constants
from sklearn.cluster import DBSCAN, KMeans

from quartz.data import Grid, MoleculeData
from scipy.spatial.distance import pdist, squareform

# SECTION - xtb property extraction functions


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
            "           -------------------------------------------------\n",
            start_idx,
        )
        for line in lines[start_idx:end_idx]:
            parts = line.split()
            props["fukui+"].append(float(parts[-3]))
            props["fukui-"].append(float(parts[-2]))
            props["fukui0"].append(float(parts[-1]))

    return props


#!SECTION
# SECTION - Calculation functions

# NOTE - overlap functions use Dice-like similarity metric


def calculate_overlap(
    grids: list[Grid], props: list[str], min_contribs: int = 10
) -> dict[str, float]:
    """ """
    prop_overlap: dict[str, float] = {}

    for key in props:
        # Convert to float64 to prevent overflow
        np_grids = [grid.npGrid[key].astype(np.float64) for grid in grids]
        combs = list(combinations(np_grids, 2))

        n_mask = np.where(
            np.count_nonzero(np_grids, axis=0) > min_contribs, True, False
        )
        np_grids[:, ~n_mask] = np.nan

        if len(combs) == 0:
            prop_overlap[key] = np.nan
            continue

        # Calculate differences pairwise to avoid huge arrays
        total_diff = 0.0
        for a, b in combs:
            total_diff += np.nansum(np.abs(a - b))

        # Sum all grids
        total_sum = sum(np.nan_to_num(np.nansum(np.abs(g))) for g in np_grids)

        # Avoid division by zero
        if total_sum == 0 or np.isinf(total_sum):
            prop_overlap[key] = np.nan
        else:
            prop_overlap[key] = np.round(1 - (total_diff / (len(combs) * total_sum)), 4)

    return prop_overlap

def dice_coefficient(reference: Grid, grids: list[Grid]) -> dict[str, dict[str, float]]:
    dice_coefs = {}
    for prop, ref_grid in reference.npGrid.items():
        dice_coefs[prop] = {}
        for g in grids:
            top = 2 * np.sum(ref_grid * g.npGrid[prop])
            bottom = np.sum(ref_grid**2) + np.sum(g.npGrid[prop]**2)
            target_id = g.mol_data.pdb_code if g.mol_data else f"target_{id(g)}"
            if bottom == 0:
                dice_coefs[prop][target_id] = np.nan
            else:
                dice_coefs[prop][target_id] = np.round(top / bottom, 4)
    return dice_coefs
        

def cosine_coefficient(reference: Grid, grids: list[Grid]) -> dict[str, dict[str, float]]:
    cosine_coefs = {}
    for prop, ref_grid in reference.npGrid.items():
        cosine_coefs[prop] = {}
        for g in grids:
            top = np.sum(ref_grid * g.npGrid[prop])
            bottom = np.sqrt(np.sum(ref_grid**2) * np.sum(g.npGrid[prop]**2))
            target_id = g.mol_data.pdb_code if g.mol_data else f"target_{id(g)}"
            if bottom == 0:
                cosine_coefs[prop][target_id] = np.nan
            else:
                cosine_coefs[prop][target_id] = np.round(top / bottom, 4)
    return cosine_coefs

def calculate_voxel_overlap(grids: list[Grid], props: list[str], metric: str = 'euclidean', min_contribs=10):
    pass


def pair_overlap(reference: Grid, targets: list[Grid]) -> dict[str, dict[str, float]]:
    overlap: dict[str, dict[str, float]] = {}

    for prop, grid in reference.npGrid.items():
        overlap[prop] = {}
        for g in targets:
            # No min_contribs as we're using a singluar grid
            np_grid = g.npGrid[prop].astype(np.float64)
            target_id = g.mol_data.pdb_code if g.mol_data else f"target_{id(g)}"

            total_diff = np.sum(np.abs(np_grid - grid))
            total_sum = np.sum(np.abs(np_grid)) + np.sum(np.abs(grid))

            if total_sum == 0 or np.isinf(total_sum):
                overlap[prop][target_id] = np.nan
            else:
                overlap[prop][target_id] = np.round(1 - (total_diff / total_sum), 4)

    return overlap

# FIXME - THIS IS TERRIBLY EXPENSIVE AND DOESNT EVEN WORK 
def all_grid_pairwise_distance(grids: list[Grid], props: list[str], absolute_vals: bool = True, metric: str = 'euclidean'):
    prop_distances = {}
    prop_distances_sum = {}
    for key in props:
        np_grids = np.array([grid.npGrid[key] for grid in grids])
        if absolute_vals:
            np_grids = np.abs(np_grids)
        prop_distances[key] = squareform(pdist(np_grids, metric))
        # prop_distances_sum[key] = 
    return prop_distances

def voxel_pair_overlap(reference: Grid, targets: list[Grid], absolute_vals: bool = True):
    overlap = {}
    pass


def calculate_average(
    grids: list[Grid], props: list[str], min_contribs: int = 10
) -> Grid:
    prop_averages: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array([grid.npGrid[key].astype(np.float64) for grid in grids])
        n_mask = np.where(
            np.count_nonzero(np_grids, axis=0) > min_contribs, True, False
        )
        np_grids[:, ~n_mask] = np.nan

        prop_averages[key] = np.nan_to_num(np.nanmean(np_grids, axis=0))

    return Grid(npGrid=prop_averages, grid_props=grids[0].grid_props)


# TODO - make use of docking score or \DeltaG = -RTlnK as the energy for boltzmann
def calculate_boltz_avg(
    grids: list[Grid],
    props: list[str],
    energy_key: str,
    beta: float = 1.0,
) -> Grid:
    """
    Function calculating Boltzmann-weighted average of grid properties. Values used for energies are specified in `energy_keys` and are retrieved from `mol_data.extra` of a `MoleculeData` object.
    energy_keys: list of keys in mol_data.extra to sum for energy calculation
    """

    prop_averages: dict[str, npt.NDArray[np.float64]] = {}
    try:
        energies = []
        valid_grids = []
        for grid in grids:
            if (
                grid.mol_data
                and grid.mol_data.extra
                and energy_key in grid.mol_data.extra
            ):
                energies.append(float(grid.mol_data.extra[energy_key]))
                valid_grids.append(grid)
            else:
                raise ValueError(
                    f"Energy key '{energy_key}' not found in mol_data.extra for grid: {grid.mol_data.filename}"
                )

        energies = np.array(energies)
        # print("Energies: ", energies)
    except Exception as e:
        raise ValueError(f"Error retrieving energies using key '{energy_key}': {e}")

    # Log-sum-exp
    max_val = np.max(-beta * energies, keepdims=True)
    exp_vals = np.exp(-beta * energies - max_val)
    partition_fn = np.sum(exp_vals)
    # print("Partition function: ", partition_fn)
    prob = exp_vals / partition_fn
    # print("Probabilities: ", prob)

    for key in props:
        scaled_grids = np.array(
            [grid.npGrid[key].astype(np.float64) * p for grid, p in zip(grids, prob)]
        )
        boltz_avg = np.mean(scaled_grids, axis=0)

        # print(f"Boltzmann average for {key}: ", boltz_avg)

        prop_averages[key] = boltz_avg

    return Grid(npGrid=prop_averages, grid_props=grids[0].grid_props)


def calculate_weighted_avg(grids: list[Grid], props: list[str]) -> Grid:
    weighted_prop_avgs: dict[str, npt.NDArray[np.float64]] = {}

    valid_pairs = []
    for grid in grids:
        if np.isnan(grid.weight):
            continue
        valid_pairs.append((grid, grid.weight))

    for key in props:
        weighted_prop_avgs[key] = np.mean(
            [grid.npGrid[key] * weight for grid, weight in valid_pairs], axis=0
        )

    return Grid(npGrid=weighted_prop_avgs, grid_props=grids[0].grid_props)


def calculate_pki_avg(
    grids: list[Grid],
    props: list[str],
    drop_nans=True,
    n_contribs: int = 10,
    pre_scaled: bool = False,
) -> Grid:
    prop_averages: dict[str, npt.NDArray[np.float64]] = {}

    valid_pairs = []
    for grid in grids:
        pki = grid.pki  # Use property
        if not np.isnan(pki):
            # print(pki)
            valid_pairs.append((grid, pki))
        elif not drop_nans:
            # print("appending nan")
            valid_pairs.append((grid, np.nan))

    if not valid_pairs:
        raise ValueError("No Grid-pKi pairs found")

    for key in props:
        if not valid_pairs:
            continue
        np_grids = np.array([grid.npGrid[key] for grid, _ in valid_pairs])
        n_mask = np.where(np.count_nonzero(np_grids, axis=0) > n_contribs, True, False)
        np_grids[:, ~n_mask] = np.nan
        if pre_scaled:
            pki_weighted_avg = np.mean([grid.npGrid[key] for grid, _ in valid_pairs], axis=0)
        else:
            pki_weighted_avg = np.mean(
                [grid.npGrid[key] * pki for grid, pki in valid_pairs], axis=0
            )
        prop_averages[key] = pki_weighted_avg

    return Grid(npGrid=prop_averages, grid_props=grids[0].grid_props)


def grid_stddev(
    grid: list[Grid],
    props: list[str],
    means: Grid | None = None,
    n_contribs: int = 10,
) -> Grid:
    prop_stddev: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array([g.npGrid[key].astype(np.float64) for g in grid])
        n_mask = np.where(np.count_nonzero(np_grids, axis=0) > n_contribs, True, False)
        np_grids[:, ~n_mask] = np.nan

        if means:
            # using np.isclose because of float magic around 0.0
            prop_stddev[key] = np.nan_to_num(
                np.nanstd(
                    np.where(np.isclose(np_grids, 0), np.nan, np_grids),
                    mean=means.npGrid[key],
                    axis=0,
                )
            )
        else:
            prop_stddev[key] = np.nan_to_num(
                np.nanstd(np.where(np.isclose(np_grids, 0), np.nan, np_grids), axis=0)
            )
    return Grid(npGrid=prop_stddev, grid_props=grid[0].grid_props)


def grid_variance(
    grid: list[Grid],
    props: list[str],
    means: Grid | None = None,
) -> Grid:
    prop_variance: dict[str, npt.NDArray[np.float64]] = {}

    for key in props:
        np_grids = np.array([g.npGrid[key].astype(np.float64) for g in grid])
        if means:
            # using np.isclose because of float magic around 0.0
            prop_variance[key] = np.nan_to_num(
                np.nanvar(
                    np.where(np.isclose(np_grids, 0), np.nan, np_grids),
                    mean=means.npGrid[key],
                    axis=0,
                )
            )
        else:
            prop_variance[key] = np.nan_to_num(
                np.nanvar(np.where(np.isclose(np_grids, 0), np.nan, np_grids), axis=0)
            )
    return Grid(npGrid=prop_variance, grid_props=grid[0].grid_props)


def apply_cutoff(
    grid: Grid,
    cutoff: dict[str, float],
    cut_type: Literal["ge", "gt", "le", "lt"],
) -> Grid:
    grid_copy = grid.npGrid.copy()
    for prop, value in cutoff.items():
        match cut_type:
            case "ge":
                grid_copy[prop][grid.npGrid[prop] >= value] = 0
            case "gt":
                grid_copy[prop][grid.npGrid[prop] > value] = 0
            case "le":
                grid_copy[prop][grid.npGrid[prop] <= value] = 0
            case "lt":
                grid_copy[prop][grid.npGrid[prop] < value] = 0

    return Grid(npGrid=grid_copy, grid_props=grid.grid_props)


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


# TODO - normalize values before clustering
def kmeans_clustering(
    grid: list[Grid], props: list[str], n_clusters: int | None = None
) -> Grid:
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

    return Grid(npGrid=clustering_results, grid_props=grid[0].grid_props)


def hdbscan_clustering(grid: list[Grid], props: list[str]):
    clustering_results: dict[str, Any] = {}

    for key in props:
        dbs = DBSCAN(eps=0.3)

    pass
