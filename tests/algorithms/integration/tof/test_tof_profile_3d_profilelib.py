from __future__ import annotations

import numpy as np
import pytest

from dials.array_family import flex
from dials_algorithms_tof_integration_ext import (
    TOFProfile3DProfilelibParams,
    fit_profile_3d_profilelib,
    profile_3d_profilelib_profile,
)

# A box of 13 x 13 pixels and 40 ToF bins of 20 us
NX, NY, NZ, X0, Y0 = 13, 13, 40, 100, 200
BIN_WIDTH = 20.0
TOF = flex.double(10000 + BIN_WIDTH * np.arange(NZ))

# alpha, beta (1/us), sigma (us), sig_x, sig_y (px), sig_p, drift_x, drift_y (px/us)
SHAPE = (0.05, 0.02, 30.0, 1.5, 1.2, 0.2, 0.0, 0.0)
POSITION = (X0 + 6.3, Y0 + 6.8, 10380.0)  # x, y (px), centroid ToF (us)
BACKGROUND = 2.0  # counts per voxel


def make_params():
    return TOFProfile3DProfilelibParams(
        {
            "alpha": 0.03,
            "alpha_min": 1e-5,
            "alpha_max": 10.0,
            "beta": 0.03,
            "beta_min": 1e-5,
            "beta_max": 10.0,
            "SigX_min": 0.2,
            "SigX_max": 20.0,
            "SigY_min": 0.2,
            "SigY_max": 20.0,
            "SigP_min": -0.95,
            "SigP_max": 0.95,
            "max_drift_factor": 1.5,
            "n_restarts": 10,
            "show_profile_failures": False,
            "library_min_i_sigma": 10.0,
            "library_min_corr": 0.5,
            "position_tolerance": 3.0,
            "free_position_min_i_sigma": 10.0,
        }
    )


def profile(shape=SHAPE, position=POSITION):
    return np.array(profile_3d_profilelib_profile(NX, NY, X0, Y0, TOF, shape, position))


def fit(counts, **options):
    """Fit background-subtracted, bin-width normalised counts, as
    dials.tof_integrate corrects them without an incident spectrum."""
    n = NX * NY * NZ
    value = (counts - BACKGROUND) / BIN_WIDTH
    return fit_profile_3d_profilelib(
        NX,
        NY,
        X0,
        Y0,
        TOF,
        flex.double(value),
        flex.double(np.full(n, 1.0 / BIN_WIDTH)),
        flex.double(np.full(n, BACKGROUND)),
        flex.double(n),
        BACKGROUND / 4000,
        make_params(),
        options,
    )


def test_profile_has_unit_sum_and_peaks_at_position():
    p = profile()
    assert p.sum() == pytest.approx(1.0)
    ix, iy, iz = np.unravel_index(np.argmax(p), (NX, NY, NZ))
    assert X0 + ix + 0.5 == pytest.approx(POSITION[0], abs=1)
    assert Y0 + iy + 0.5 == pytest.approx(POSITION[1], abs=1)


def test_shape_fit_recovers_a_strong_peak():
    rng = np.random.default_rng(1)
    counts = rng.poisson(BACKGROUND + 3000 * profile())
    result = fit(counts)
    assert result["success"]
    alpha, beta, sigma, sig_x, sig_y, sig_p, drift_x, drift_y = result["shape"]
    assert sig_x == pytest.approx(SHAPE[3], rel=0.1)
    assert sig_y == pytest.approx(SHAPE[4], rel=0.1)
    assert sigma == pytest.approx(SHAPE[2], rel=0.2)
    x, y, t = result["position"]
    assert x == pytest.approx(POSITION[0], abs=0.2)
    assert y == pytest.approx(POSITION[1], abs=0.2)
    assert t == pytest.approx(POSITION[2], abs=10)
    intensity = result["intensity"] * BIN_WIDTH
    sigma_i = np.sqrt(result["variance"]) * BIN_WIDTH
    assert abs(intensity - 3000) < 4 * sigma_i
    assert result["chi2"] == pytest.approx(1.0, abs=0.2)


def test_shape_fit_refuses_a_box_of_noise():
    rng = np.random.default_rng(2)
    counts = rng.poisson(np.full(NX * NY * NZ, BACKGROUND))
    assert not fit(counts)["success"]


def _fixed_fits(n_counts, n_trials, scatter, seed):
    """Weak peaks scattered about their expected position, fitted there with
    the shape broadened by that scatter."""
    rng = np.random.default_rng(seed)
    values, sigmas = [], []
    for _ in range(n_trials):
        true_position = tuple(
            np.array(POSITION) + rng.normal(0.0, 1.0, 3) * np.array(scatter)
        )
        counts = rng.poisson(BACKGROUND + n_counts * profile(position=true_position))
        result = fit(
            counts, shape=SHAPE, seed=POSITION, fix_position=True, scatter=scatter
        )
        assert result["success"]
        values.append(result["intensity"] * BIN_WIDTH)
        sigmas.append(np.sqrt(result["variance"]) * BIN_WIDTH)
    return np.array(values), np.array(sigmas)


def test_fixed_position_fit_is_unbiased_on_weak_peaks():
    values, sigmas = _fixed_fits(30, 400, (0.7, 0.7, 15.0), seed=3)
    sem = values.std() / np.sqrt(len(values))
    assert abs(values.mean() - 30) < 4 * sem
    assert ((values - 30) / sigmas).std() == pytest.approx(1.0, abs=0.15)


def test_fixed_position_fit_reads_zero_for_no_peak():
    # Not constrained to be positive, so with no peak the mean is zero
    values, sigmas = _fixed_fits(0, 400, (0.7, 0.7, 15.0), seed=4)
    sem = values.std() / np.sqrt(len(values))
    assert abs(values.mean()) < 4 * sem
    assert (values < 0).sum() > 100


def test_free_position_reads_weak_peaks_high():
    # Why a weak reflection's position is held fixed: fitted, it finds the
    # brightest noise
    rng = np.random.default_rng(5)
    scatter = np.array((0.7, 0.7, 15.0))
    values = []
    for _ in range(200):
        true_position = tuple(np.array(POSITION) + rng.normal(0.0, 1.0, 3) * scatter)
        counts = rng.poisson(BACKGROUND + 30 * profile(position=true_position))
        result = fit(counts, shape=SHAPE, seed=POSITION, tolerance=tuple(3 * scatter))
        assert result["success"]
        values.append(result["intensity"] * BIN_WIDTH)
    assert np.mean(values) > 1.1 * 30


def test_free_position_fit_is_unbiased_and_calibrated_on_strong_peaks():
    rng = np.random.default_rng(6)
    scatter = np.array((0.7, 0.7, 15.0))
    values, sigmas = [], []
    for _ in range(150):
        true_position = tuple(np.array(POSITION) + rng.normal(0.0, 1.0, 3) * scatter)
        counts = rng.poisson(BACKGROUND + 1000 * profile(position=true_position))
        result = fit(counts, shape=SHAPE, seed=POSITION, tolerance=tuple(3 * scatter))
        assert result["success"]
        values.append(result["intensity"] * BIN_WIDTH)
        sigmas.append(np.sqrt(result["variance"]) * BIN_WIDTH)
    values, sigmas = np.array(values), np.array(sigmas)
    sem = values.std() / np.sqrt(len(values))
    assert abs(values.mean() - 1000) < 4 * sem
    assert ((values - 1000) / sigmas).std() == pytest.approx(1.0, abs=0.2)


def test_drift_is_recovered():
    # About 1.5 px across the width of the peak in ToF
    rng = np.random.default_rng(7)
    shape = SHAPE[:6] + (0.01, -0.008)
    counts = rng.poisson(BACKGROUND + 5000 * profile(shape=shape))
    result = fit(counts)
    assert result["success"]
    assert result["shape"][6] == pytest.approx(0.01, abs=0.002)
    assert result["shape"][7] == pytest.approx(-0.008, abs=0.002)
