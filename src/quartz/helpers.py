from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from rdkit.Chem.rdmolops import GetFormalCharge
from scipy.stats import pearsonr

from quartz.data import Grid
from quartz.dataprocessing import dice_coefficient
from quartz.grids import to_dxgrid


def _read_first_n_floats(lines: list[str], nvals: int) -> np.ndarray:
    vals: list[float] = []
    for line in lines:
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue
            if len(vals) == nvals:
                return np.asarray(vals, dtype=np.float32)
    raise ValueError(f"Could not parse {nvals} floats from DX data section")


def load_dx_clusters(
    template_grid: Grid,
    prop: str,
    masks_glob: str = "grids/data_aggl_clusters_polarizability_label*.dx",
    threshold: float = 0.5,
    strict: bool = True,
) -> Grid | None:
    """Reconstruct a labeled cluster grid from one-hot DX mask files."""
    mask_paths = sorted(Path(".").glob(masks_glob))
    if not mask_paths:
        raise FileNotFoundError(f"No files matched: {masks_glob}")

    shape_ref = tuple(template_grid.npGrid[prop].shape)
    origin_ref = tuple(float(x) for x in template_grid.grid_props["origin"])
    size_ref = float(template_grid.grid_props["size"])

    labels = np.full(shape_ref, -1, dtype=np.int32)

    for dx_path in mask_paths:
        counts = None
        origin = None
        deltas = []

        with dx_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        data_start = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("object 1 class gridpositions counts"):
                parts = s.split()
                counts = tuple(int(x) for x in parts[-3:])
            elif s.startswith("origin"):
                parts = s.split()
                origin = tuple(float(x) for x in parts[1:4])
            elif s.startswith("delta"):
                parts = s.split()
                deltas.append(tuple(float(x) for x in parts[1:4]))
            elif "data follows" in s:
                data_start = i + 1
                break

        if counts is None or origin is None or data_start is None or len(deltas) < 3:
            raise ValueError(f"Malformed DX header in {dx_path}")

        # Axis-aligned spacing from first 3 delta vectors.
        spacing = tuple(max(abs(v) for v in d) for d in deltas[:3])

        msg = None
        if counts != shape_ref:
            msg = (
                f"Shape mismatch for {dx_path.name}: dx={counts}, template={shape_ref}"
            )
        elif not np.allclose(origin, origin_ref, atol=1e-6):
            msg = f"Origin mismatch for {dx_path.name}: dx={origin}, template={origin_ref}"
        elif not np.allclose(spacing, (size_ref, size_ref, size_ref), atol=1e-6):
            msg = f"Spacing mismatch for {dx_path.name}: dx={spacing}, template={size_ref}"

        if msg is not None:
            if strict:
                raise ValueError(msg)
            print(msg)
            return None

        nvals = int(np.prod(shape_ref))
        data = _read_first_n_floats(lines[data_start:], nvals)
        mask = data.reshape(shape_ref)
        label_id = int(dx_path.stem.split("label")[-1])
        labels[mask > threshold] = label_id

    return Grid(npGrid={prop: labels}, grid_props=template_grid.grid_props)


def get_overlap_df(cluster_grid, ligand_grids, avg_grid, props):
    for key in props:
        pass


def plot_cluster_overlap(
    cluster_grid,
    ligand_grids,
    avg_grid,
    props,
    weight_power=1,
    clusters_plotted=None,
    title=None,
    color: str | None = "doi",
    save_img: bool = False,
    img_dir: str | Path = "",
):
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    if title is None:
        _title = "auto"
    for key in props:
        overlap_per_cluster: dict[str, list[float]] = {}
        labels = cluster_grid.npGrid[key]

        if clusters_plotted is None:
            clusters_plotted = range(np.min(labels), np.max(labels) + 1)

        for i in clusters_plotted:
            indices = labels == i

            #  * (g.pki**weight_power)
            masked_grids = [g.npGrid[key][indices] for g in ligand_grids]
            if not masked_grids:
                continue

            masked_grids_obj = [
                Grid(npGrid={key: m}, grid_props=g.grid_props)
                for g, m in zip(ligand_grids, masked_grids)
            ]

            masked_avg = avg_grid.npGrid[key][indices]
            masked_avg_obj = Grid(
                npGrid={key: masked_avg}, grid_props=ligand_grids[0].grid_props
            )

            pki_dices = dice_coefficient(masked_avg_obj, masked_grids_obj)
            overlap_per_cluster[f"pki_dice_{i}"] = list(pki_dices[key].values())

        df = pd.DataFrame(overlap_per_cluster)
        df["pki"] = [g.pki for g in ligand_grids]
        df["code"] = [g.mol_data.pdb_code for g in ligand_grids]
        df["doi"] = [g.mol_data.extra["doi"] for g in ligand_grids]
        df["charge"] = [str(GetFormalCharge(g.mol_data.rdmol)) for g in ligand_grids]
        df.dropna(inplace=True, axis=1)

        df.to_csv(f"overlaps_df_{key}.csv")

        pki_cols = [col for col in df.columns if col.startswith("pki_dice_")]

        for cluster_col in pki_cols:
            # corr = np.round(df[cluster_col].corr(df["pki"]), 3)
            corr = np.round(pearsonr(df[cluster_col], df["pki"]).statistic, 3)
            cluster_id = cluster_col.split("_")[-1]

            if _title and _title == "auto":
                title = f"Dice with averaged pKi grids ({key}) | Cluster {cluster_id} | Corr: {corr}"

            fig = px.scatter(
                df,
                x=cluster_col,
                y="pki",
                hover_data=["code"],
                title=title,
                labels={
                    cluster_col: f"Dice with cluster {cluster_id}",
                    "pki": "pKi",
                },
                color=color,
                width=1500,
                height=900,
            )
            if save_img and img_dir:
                fig.write_image(img_dir / f"{key}_dice_overlap_cluster{cluster_id}.svg")
            fig.show()


def export_cluster_masks(
    cluster_grid: Grid, prop: str, prefix: str, background_label: int = -1
) -> None:
    labels = cluster_grid.npGrid[prop]
    for label_id in sorted(np.unique(labels)):
        if label_id == background_label:
            continue
        mask = np.zeros_like(labels, dtype=np.float32)
        mask[labels == label_id] = 1.0
        to_dxgrid(
            mask,
            origin=cluster_grid.grid_props["origin"],
            spacing=cluster_grid.grid_props["size"],
            filename=f"{prefix}_{prop}_label{label_id}.dx",
        )


def export_pki_cluster_masks(cluster_grid: Grid, pki_grid: Grid, prop):
    labels = cluster_grid.npGrid[prop]
    pki_npgrid = pki_grid.npGrid[prop]
    for label_id in sorted(np.unique(labels)):
        mask = np.zeros_like(labels, dtype=np.float32)
        mask[labels == label_id] = pki_npgrid[labels == label_id]

        to_dxgrid(
            mask,
            origin=cluster_grid.grid_props["origin"],
            spacing=cluster_grid.grid_props["size"],
            filename=f"pkiavg_{prop}_label{label_id}.dx",
        )


def combs_overlap(
    n_clusters: int,
    comb_len: int,
    props: list[str],
    cluster_grid: Grid,
    avg_grid: Grid,
    mol_grids: list[Grid],
):
    combs_overlap = {}
    combs = combinations(range(n_clusters), comb_len)

    for comb in combs:
        for key in props:
            labels = cluster_grid.npGrid[key]
            indices = np.isin(labels, comb)
            masked_mol_grids = [
                Grid(
                    npGrid={key: g.npGrid[key][indices]},
                    grid_props=g.grid_props,
                )
                for g in mol_grids
            ]
            masked_ref_grid = Grid(
                npGrid={key: avg_grid.npGrid[key][indices]},
                grid_props=avg_grid.grid_props,
            )
            dices = dice_coefficient(masked_ref_grid, masked_mol_grids)
            combs_overlap[str(comb)] = list(dices[key].values())

    overlap_df = pd.DataFrame(combs_overlap)
    overlap_df["pKi"] = [g.pki for g in mol_grids]

    corr_rows = []
    for comb_col in overlap_df.columns:
        if comb_col == "pKi":
            continue

        corr_val = overlap_df[comb_col].corr(overlap_df["pKi"])

        comb_ids = [x.strip() for x in comb_col.strip("()").split(",") if x.strip()]
        comb_label = "+".join(str(int(x)) for x in comb_ids)

        corr_rows.append(
            {
                "cluster_combination": comb_label,
                "corr_with_pKi": np.round(corr_val, 3),
            }
        )

    combs_overlap = (
        pd.DataFrame(corr_rows)
        .sort_values("corr_with_pKi", ascending=False)
        .reset_index(drop=True)
    )

    return combs_overlap


def plot_voxel_clusters(
    cluster_grid: Grid,
    prop: str,
    max_points: int = 50_000,
    background_label: int = -1,
):
    coords = np.argwhere(np.ones_like(cluster_grid.npGrid[prop], dtype=bool))
    label_values = cluster_grid.npGrid[prop][tuple(coords.T)]

    if len(coords) > max_points:
        sampled = np.random.default_rng(42).choice(
            len(coords), size=max_points, replace=False
        )
        coords = coords[sampled]
        label_values = label_values[sampled]

    fig = px.scatter_3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        color=label_values.astype(str),
        title=f"Clustering for {prop}",
        labels={"x": "X", "y": "Y", "z": "Z", "color": "Cluster"},
        opacity=0.55,
        width=1500,
        height=900,
    )
    fig.update_traces(marker={"size": 2})
    fig.show()


def comb_clusters_overlap(
    cluster_grid,
    mol_grids,
    avg_grid,
    props,
    clusters_to_combine: list[list[int]],
):
    """
    Calculates Dice coefficients for multiple arbitrary sets of combined clusters
    and the corresponding non-scaled grids.
    """
    dfs = {}

    for key in props:
        df = pd.DataFrame()
        df["pKi"] = [g.pki for g in mol_grids]
        df["code"] = [g.mol_data.pdb_code for g in mol_grids]
        df["doi"] = [g.mol_data.extra.get("doi", "missing") for g in mol_grids]

        for cluster_combo in clusters_to_combine:
            combined_label = "+".join(str(c) for c in cluster_combo)

            # mask the specified clusters
            labels = cluster_grid.npGrid[key]
            indices = np.isin(labels, cluster_combo)

            # mask the grids
            masked_grids = [g.npGrid[key][indices] for g in mol_grids]
            masekd_grids_obj = [
                Grid(npGrid={key: m}, grid_props=g.grid_props)
                for g, m in zip(mol_grids, masked_grids)
            ]

            # compare against weighted average grid
            pki_prop = avg_grid.npGrid[key]
            masked_pki = pki_prop[indices]
            masked_pki_obj = Grid(
                npGrid={key: masked_pki}, grid_props=mol_grids[0].grid_props
            )

            pki_dices = dice_coefficient(masked_pki_obj, masekd_grids_obj)

            df[f"pki_dice_{combined_label}"] = list(pki_dices[key].values())

        df.dropna(inplace=True)
        dfs[key] = df

    return dfs


def plot_comb_clusters_overlap(
    dfs,
    color: str | None = None,
    save_img: bool = False,
    img_dir: str | Path = "",
):
    """
    Plots the overlaps calculated by comb_clusters_overlap and returns
    a dataframe of the per-DOI correlations.
    """
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
    corr_per_doi_rows = []

    for prop, df in dfs.items():
        pki_cols = [col for col in df.columns if col.startswith("pki_dice_")]

        for cluster_col in pki_cols:
            combined_label = cluster_col.replace("pki_dice_", "")
            corr = df[cluster_col].corr(df["pKi"]).round(2)
            fig = px.scatter(
                df,
                x=cluster_col,
                y="pKi",
                hover_data=["code"],
                title=f"Dice with averaged pKi grids ({prop}) | Corr: {corr}",
                labels={
                    cluster_col: f"Dice with combined clusters [{combined_label}]",
                    "pKi": "pKi",
                },
                color=color,
                width=1500,
                height=900,
            )
            if save_img and img_dir:
                fig.write_image(
                    img_dir / f"{prop}_dice_overlap_clusters{combined_label}.svg"
                )
            fig.show()

            doi_counts = df.groupby("doi").size()
            valid_dois = doi_counts[doi_counts > 5].index

            for doi in valid_dois:
                doi_df = df[df["doi"] == doi]
                if doi_df[cluster_col].nunique() < 2 or doi_df["pKi"].nunique() < 2:
                    doi_corr = np.nan
                else:
                    doi_corr = doi_df[cluster_col].corr(doi_df["pKi"])

                corr_per_doi_rows.append(
                    {
                        "property": prop,
                        "cluster_col": cluster_col,
                        "doi": doi,
                        "n_compounds": int(len(doi_df)),
                        "corr": doi_corr,
                    }
                )

    corr_per_doi = pd.DataFrame(corr_per_doi_rows)
    if not corr_per_doi.empty:
        corr_per_doi = corr_per_doi.sort_values(
            by=["property", "cluster_col", "n_compounds"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return corr_per_doi
