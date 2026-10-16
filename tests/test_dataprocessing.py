"""Unit tests for dice, pKi scaling, averaging and clustering used by clustering.ipynb."""

import numpy as np
import pytest

from molecule_factory import make_dummy_grid
from quartz.data import pki_grid_scale, set_weights
from quartz.dataprocessing import (
    calculate_pki_avg,
    calculate_weighted_avg,
    dice_coefficient,
    kmeans_clustering,
    spatial_agglomerative_clustering,
)


def test_dice_matches_formula():
    a = make_dummy_grid(np.array([1.0, 2.0, 0.0]), "A")
    b = make_dummy_grid(np.array([2.0, 2.0, 0.0]), "B")
    # 2*(1*2 + 2*2) / (1 + 4 + 4 + 4) = 12 / 13
    result = dice_coefficient(a, [b])["polarizability"]["B"]
    assert result == pytest.approx(12 / 13, abs=1e-4)


def test_dice_identical_is_one():
    a = make_dummy_grid(np.array([0.3, 1.2, 0.0, 2.0]), "A")
    b = make_dummy_grid(np.array([0.3, 1.2, 0.0, 2.0]), "B")
    assert dice_coefficient(a, [b])["polarizability"]["B"] == pytest.approx(1.0)


def test_dice_disjoint_support_is_zero():
    a = make_dummy_grid(np.array([1.0, 0.0]), "A")
    b = make_dummy_grid(np.array([0.0, 1.0]), "B")
    assert dice_coefficient(a, [b])["polarizability"]["B"] == pytest.approx(0.0)


def test_dice_is_symmetric():
    a = make_dummy_grid(np.array([1.0, 2.0, 0.5]), "A")
    b = make_dummy_grid(np.array([0.2, 1.0, 3.0]), "B")
    ab = dice_coefficient(a, [b])["polarizability"]["B"]
    ba = dice_coefficient(b, [a])["polarizability"]["A"]
    assert ab == pytest.approx(ba)


def test_dice_zero_grid_gives_nan():
    a = make_dummy_grid(np.zeros(3), "A")
    b = make_dummy_grid(np.zeros(3), "B")
    assert np.isnan(dice_coefficient(a, [b])["polarizability"]["B"])


def test_pki_grid_scale_multiplies_by_pki():
    g = make_dummy_grid(np.array([1.0, 2.0]), "A", pki=6.0)
    scaled = pki_grid_scale(g)
    np.testing.assert_allclose(scaled.npGrid["polarizability"], [6.0, 12.0])
    assert scaled.grid_props == g.grid_props


def test_pki_grid_scale_returns_none_without_pki():
    g = make_dummy_grid(np.array([1.0, 2.0]), "A", pki=None)
    assert pki_grid_scale(g) is None


def test_set_weights_normalises_pki_and_leaves_grids_untouched():
    grids = [
        make_dummy_grid(np.array([1.0, 1.0]), "A", pki=5.0),
        make_dummy_grid(np.array([1.0, 1.0]), "B", pki=10.0),
    ]
    set_weights(grids, "pki")
    assert grids[0].weight == pytest.approx(5 / 15)
    assert grids[1].weight == pytest.approx(10 / 15)
    np.testing.assert_allclose(grids[0].npGrid["polarizability"], [1.0, 1.0])


def test_pki_avg_is_mean_of_pki_scaled_grids():
    grids = [
        make_dummy_grid(np.array([1.0, 0.0, 2.0]), "A", pki=5.0),
        make_dummy_grid(np.array([0.0, 3.0, 2.0]), "B", pki=7.0),
    ]
    avg = calculate_pki_avg(grids, props=["polarizability"], n_contribs=0)
    expected = (np.array([5.0, 0.0, 10.0]) + np.array([0.0, 21.0, 14.0])) / 2
    np.testing.assert_allclose(avg.npGrid["polarizability"], expected)


def test_pki_avg_pre_scaled_uses_grid_values_directly():
    grids = [
        make_dummy_grid(np.array([1.0, 2.0]), "A", pki=5.0),
        make_dummy_grid(np.array([3.0, 4.0]), "B", pki=7.0),
    ]
    avg = calculate_pki_avg(
        grids, props=["polarizability"], n_contribs=0, pre_scaled=True
    )
    np.testing.assert_allclose(avg.npGrid["polarizability"], [2.0, 3.0])


def test_pki_avg_skips_grids_without_pki():
    grids = [
        make_dummy_grid(np.array([1.0, 1.0]), "A", pki=4.0),
        make_dummy_grid(np.array([9.0, 9.0]), "B", pki=None),
    ]
    avg = calculate_pki_avg(grids, props=["polarizability"], n_contribs=0)
    np.testing.assert_allclose(avg.npGrid["polarizability"], [4.0, 4.0])


def test_pki_avg_without_any_pki_raises():
    grids = [make_dummy_grid(np.array([1.0]), "A", pki=None)]
    with pytest.raises(ValueError):
        calculate_pki_avg(grids, props=["polarizability"])


@pytest.mark.xfail(
    strict=True,
    reason="calculate_pki_avg computes the n_contribs mask but never applies it",
)
def test_pki_avg_n_contribs_masks_sparse_voxels():
    grids = [
        make_dummy_grid(np.array([1.0, 1.0]), "A", pki=5.0),
        make_dummy_grid(np.array([0.0, 1.0]), "B", pki=5.0),
    ]
    # voxel 0 is contributed by one molecule only, so with n_contribs=1 it is dropped
    avg = calculate_pki_avg(grids, props=["polarizability"], n_contribs=1)
    assert avg.npGrid["polarizability"][0] == 0.0


def test_weighted_avg_uses_extra_weight():
    grids = [
        make_dummy_grid(np.array([2.0, 0.0]), "A", pki=5.0),
        make_dummy_grid(np.array([0.0, 4.0]), "B", pki=5.0),
    ]
    grids[0].extra["weight"] = 0.25
    grids[1].extra["weight"] = 0.75
    avg = calculate_weighted_avg(grids, props=["polarizability"])
    # mean over grids of (grid * weight); not normalised by the sum of weights
    np.testing.assert_allclose(avg.npGrid["polarizability"], [0.25, 1.5])


def _halves(value_scale: float):
    left = np.zeros((4, 4, 4))
    left[:2] = value_scale
    return left


def test_kmeans_separates_two_spatial_halves():
    grids = [
        make_dummy_grid(_halves(1.0), "A", pki=5.0),
        make_dummy_grid(_halves(2.0), "B", pki=5.0),
    ]
    labels = kmeans_clustering(grids, props=["polarizability"], n_clusters=2).npGrid[
        "polarizability"
    ]
    assert labels.shape == (4, 4, 4)
    assert labels.dtype == np.int32
    assert len(np.unique(labels[:2])) == 1
    assert len(np.unique(labels[2:])) == 1
    assert labels[0, 0, 0] != labels[3, 0, 0]


def test_spatial_clustering_separates_two_spatial_halves():
    grids = [
        make_dummy_grid(_halves(1.0), "A", pki=5.0),
        make_dummy_grid(_halves(2.0), "B", pki=5.0),
    ]
    labels = spatial_agglomerative_clustering(grids, ["polarizability"], n_clusters=2)
    labels = labels.npGrid["polarizability"]
    assert labels.shape == (4, 4, 4)
    assert len(np.unique(labels[:2])) == 1
    assert len(np.unique(labels[2:])) == 1
    assert labels[0, 0, 0] != labels[3, 0, 0]


def test_spatial_clustering_single_grid_runs_on_average_grid():
    # clusters the averaged grid alone, as clustering.ipynb does for pki_avg
    grid = make_dummy_grid(_halves(1.0), "AVG")
    labels = spatial_agglomerative_clustering([grid], ["polarizability"], 2)
    assert labels.npGrid["polarizability"].shape == (4, 4, 4)
