"""Regression checks for the diagnostics and directed neighbor lookup."""

from statistics import NormalDist

import numpy as np

from looped_mechanisms.design import HIDDEN
from looped_mechanisms.geometry import NEIGHBORS, gaussianity, layer_neighbors


def test_cross_loop_neighbors_exclude_every_token_of_source_prompt() -> None:
    rng = np.random.default_rng(5)
    states = rng.standard_normal((3, 8, 6)).astype(np.float32)
    prompts = np.repeat(np.arange(4), 2)
    excluded = prompts[:, None] == prompts[None, :]
    states[1, 1] = states[0, 0]  # The identical vector is from the same prompt.

    results, means = layer_neighbors(np, states, excluded)

    for source in range(3):
        for target in range(3):
            if source == target:
                assert np.isnan(means[source, target])
                continue
            distances = np.linalg.norm(
                states[source, :, None].astype(np.float64) - states[target, None],
                axis=-1,
            ) / np.sqrt(HIDDEN)
            distances[excluded] = np.inf
            closest = np.argsort(distances, axis=1)[:, :NEIGHBORS]
            expected = np.take_along_axis(distances, closest, axis=1)
            np.testing.assert_array_equal(results[source, target]["index"], closest)
            np.testing.assert_allclose(results[source, target]["distance"], expected, atol=1e-7)
            np.testing.assert_allclose(means[source, target], expected[:, 0].mean(), atol=1e-7)


def test_gaussianity_detects_non_normal_activations() -> None:
    rng = np.random.default_rng(31)
    normal = NormalDist()
    count = 2048
    quantiles = np.array([normal.inv_cdf((i + 0.5) / count) for i in range(count)])
    projections = np.eye(4, dtype=np.float32)
    gaussian = gaussianity(
        rng.standard_normal((count, 4)).astype(np.float32), projections, quantiles
    )
    skewed = gaussianity(
        rng.exponential(size=(count, 4)).astype(np.float32), projections, quantiles
    )
    assert gaussian["qq_rmse"] < 0.06 < skewed["qq_rmse"]
    assert gaussian["median_abs_excess_kurtosis"] < 0.4 < skewed["median_abs_excess_kurtosis"]
