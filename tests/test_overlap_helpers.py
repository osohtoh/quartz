"""Cluster-overlap helpers used by clustering.ipynb, run on substituted benzenes."""

import numpy as np
import pytest
from gridData import Grid as DXGrid

from quartz.data import Grid
from quartz.dataprocessing import calculate_pki_avg, spatial_agglomerative_clustering
from quartz.helpers import (
    comb_clusters_overlap,
    combs_overlap,
    export_cluster_masks,
    plot_cluster_overlap,
    plot_comb_clusters_overlap,
    plot_voxel_clusters,
)

PROP = "polarizability"
N_CLUSTERS = 3


@pytest.fixture(scope="module")
def pipeline(substituted_grids):
    mols = list(substituted_grids.values())
    clusters = spatial_agglomerative_clustering(mols, [PROP], n_clusters=N_CLUSTERS)
    avg = calculate_pki_avg(mols, props=[PROP], n_contribs=0)
    return mols, clusters, avg


def _manual_dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * np.sum(a * b) / (np.sum(a * a) + np.sum(b * b))


def test_combs_overlap_returns_one_row_per_combination(pipeline):
    mols, clusters, avg = pipeline
    result = combs_overlap(N_CLUSTERS, 1, [PROP], clusters, avg, mols)
    assert len(result) == N_CLUSTERS
    assert set(result["cluster_combination"]) == {"0", "1", "2"}
    assert result["corr_with_pKi"].is_monotonic_decreasing


def test_combs_overlap_pair_combinations(pipeline):
    mols, clusters, avg = pipeline
    result = combs_overlap(N_CLUSTERS, 2, [PROP], clusters, avg, mols)
    assert set(result["cluster_combination"]) == {"0+1", "0+2", "1+2"}


def test_combs_overlap_correlation_matches_manual_dice(pipeline):
    mols, clusters, avg = pipeline
    labels = clusters.npGrid[PROP]
    pki = np.array([g.pki for g in mols])
    result = combs_overlap(N_CLUSTERS, 1, [PROP], clusters, avg, mols)

    for _, row in result.iterrows():
        cid = int(row["cluster_combination"])
        mask = labels == cid
        ref = avg.npGrid[PROP][mask]
        dices = np.array([_manual_dice(ref, g.npGrid[PROP][mask]) for g in mols])
        expected = np.corrcoef(dices, pki)[0, 1]
        assert row["corr_with_pKi"] == pytest.approx(expected, abs=2e-3)


def test_comb_clusters_overlap_columns_and_values(pipeline):
    mols, clusters, avg = pipeline
    labels = clusters.npGrid[PROP]
    combos = [[0, 1], [2]]
    dfs = comb_clusters_overlap(clusters, mols, avg, [PROP], combos)

    df = dfs[PROP]
    assert list(df["code"]) == [g.mol_data.pdb_code for g in mols]
    assert set(df.columns) >= {"pKi", "code", "doi", "pki_dice_0+1", "pki_dice_2"}

    for combo, col in zip(combos, ["pki_dice_0+1", "pki_dice_2"]):
        mask = np.isin(labels, combo)
        ref = avg.npGrid[PROP][mask]
        expected = [_manual_dice(ref, g.npGrid[PROP][mask]) for g in mols]
        np.testing.assert_allclose(df[col].to_numpy(), expected, atol=1e-4)


def test_export_cluster_masks_round_trip(pipeline, tmp_path, monkeypatch):
    mols, clusters, _ = pipeline
    monkeypatch.chdir(tmp_path)

    export_cluster_masks(clusters, PROP, "aggl")

    labels = clusters.npGrid[PROP]
    for cid in np.unique(labels):
        path = tmp_path / "grids" / f"data_aggl_{PROP}_label{cid}.dx"
        assert path.exists(), path
        dx = DXGrid(path)
        np.testing.assert_allclose(dx.grid, (labels == cid).astype(np.float32))
        np.testing.assert_allclose(dx.origin, clusters.grid_props["origin"])


def test_export_cluster_masks_skips_background_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    labels = np.array([[[-1, 0], [1, 1]]], dtype=np.int32)
    cluster_grid = Grid(
        npGrid={PROP: labels},
        grid_props={"origin": (0.0, 0.0, 0.0), "size": 1.0},
    )
    export_cluster_masks(cluster_grid, PROP, "bg", background_label=-1)
    names = {p.name for p in (tmp_path / "grids").iterdir()}
    assert names == {f"data_bg_{PROP}_label0.dx", f"data_bg_{PROP}_label1.dx"}


def test_plot_cluster_overlap_runs(pipeline, no_show, tmp_path, monkeypatch):
    mols, clusters, avg = pipeline
    monkeypatch.chdir(tmp_path)
    plot_cluster_overlap(clusters, mols, avg, [PROP], 0)
    assert (tmp_path / f"overlaps_df_{PROP}.csv").exists()


@pytest.mark.xfail(
    strict=True,
    raises=NameError,
    reason="plot_cluster_overlap only defines _title when title is None",
)
def test_plot_cluster_overlap_accepts_explicit_title(
    pipeline, no_show, tmp_path, monkeypatch
):
    mols, clusters, avg = pipeline
    monkeypatch.chdir(tmp_path)
    plot_cluster_overlap(clusters, mols, avg, [PROP], 0, title="custom")


def test_plot_voxel_clusters_runs(pipeline, no_show):
    _, clusters, _ = pipeline
    plot_voxel_clusters(clusters, PROP, max_points=100)


def test_plot_comb_clusters_overlap_reports_per_doi_correlations(pipeline, no_show):
    mols, clusters, avg = pipeline
    dfs = comb_clusters_overlap(clusters, mols, avg, [PROP], [[0], [1, 2]])
    corr = plot_comb_clusters_overlap(dfs)
    assert not corr.empty
    assert set(corr["property"]) == {PROP}
    assert set(corr["cluster_col"]) == {"pki_dice_0", "pki_dice_1+2"}
    assert (corr["n_compounds"] > 5).all()
