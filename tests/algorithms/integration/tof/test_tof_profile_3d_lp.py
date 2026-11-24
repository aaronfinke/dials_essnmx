"""
Tests for the long-pulse ToF profile (profile_3d_lp): the closed-form line
shape against a numerical convolution, its numerical limits, and round-trip
fits on synthetic long-pulse shoeboxes.
"""

from __future__ import annotations

import numpy as np
import pytest

from dials.array_family import flex
from dials.command_line.tof_integrate import phil_scope
from dials_algorithms_tof_integration_ext import (
    TOFProfile3DICParams,
    TOFProfile3DLPParams,
    fit_profile_3d_ic,
    fit_profile_3d_lp,
    fit_profile_3d_lp_library,
    tof_emg_cdf,
    tof_long_pulse_line_shape,
)

# ESS-like long pulse: plateau, moderator emission and Gaussian broadening (us)
W_ESS = 2860.0
TAU = 90.0
SIGMA = 297.0


def line_shape(t, T0=0.0, W=W_ESS, tau=TAU, sigma=SIGMA):
    return np.array(tof_long_pulse_line_shape(flex.double(t), T0, W, tau, sigma))


def lp_params(**kwargs):
    p = phil_scope.extract().profile_3d_lp
    d = {
        "pulse_width": W_ESS,
        "optimize_pulse_width": p.optimize_pulse_width,
        "pulse_width_min_factor": p.min_pulse_width_factor,
        "pulse_width_max_factor": p.max_pulse_width_factor,
        "tau": TAU,
        "tau_min": p.min_tau,
        "tau_max": p.max_tau,
        "optimize_emission": p.optimize_emission,
        "sigma": p.init_sigma,
        "sigma_min": p.min_sigma,
        "sigma_max": p.max_sigma,
        "SigX_min": p.min_sig_x,
        "SigX_max": p.max_sig_x,
        "SigY_min": p.min_sig_y,
        "SigY_max": p.max_sig_y,
        "SigP": p.init_sig_p,
        "SigP_min": p.min_sig_p,
        "SigP_max": p.max_sig_p,
        "n_restarts": p.n_restarts,
        "optimize_profile": True,
        "max_drift_factor": p.max_drift_factor,
        "show_profile_failures": False,
    }
    d.update(kwargs)
    return TOFProfile3DLPParams(d)


def ic_params():
    p = phil_scope.extract().profile_3d_ic
    return TOFProfile3DICParams(
        {
            "A": p.init_A,
            "A_min": p.min_A,
            "A_max": p.max_A,
            "B": p.init_B,
            "B_min": p.min_B,
            "B_max": p.max_B,
            "R": p.init_R,
            "R_min": p.min_R,
            "R_max": p.max_R,
            "SigX_min": p.min_sig_x,
            "SigX_max": p.max_sig_x,
            "SigY_min": p.min_sig_y,
            "SigY_max": p.max_sig_y,
            "SigP": p.init_sig_p,
            "SigP_min": p.min_sig_p,
            "SigP_max": p.max_sig_p,
            "HatWidth": p.hat_width,
            "KConv": p.kconv,
            "n_restarts": p.n_restarts,
            "optimize_profile": True,
            "max_drift_factor": p.max_drift_factor,
            "show_profile_failures": False,
        }
    )


def make_shoebox(
    rng,
    I_true=20000.0,
    bg=2.0,
    nx=11,
    ny=11,
    bin_us=100.0,
    nz=70,
    T0=1500.0,
    W=W_ESS,
    tau=TAU,
    sigma=SIGMA,
    sx=1.5,
    sy=1.2,
    rho=0.2,
    gain=1.0,
):
    """
    A background-subtracted Poisson shoebox of a long-pulse peak, laid out
    as integrate_shoebox fills it: coords (x, y, z) are pixel/frame centres,
    tof the ToF bin centres (us). gain scales counts to corrected units, as
    the Lorentz and spectrum corrections do. Returns the arrays, the per-voxel
    gains and the true (corrected) intensity inside the box.
    """
    tof = 95000.0 + bin_us * (np.arange(nz) + 0.5)
    shape = line_shape(tof - tof[0], T0, W, tau, sigma) * bin_us
    xs = np.arange(nx) + 0.5
    ys = np.arange(ny) + 0.5
    X, Y = np.meshgrid(xs - xs.mean(), ys - ys.mean(), indexing="ij")
    q = X**2 / sx**2 + Y**2 / sy**2 - 2 * rho * X * Y / (sx * sy)
    bvg = np.exp(-q / (2 * (1 - rho**2)))
    bvg /= bvg.sum()
    model = I_true * bvg[:, :, None] * shape[None, None, :]
    obs = rng.poisson(model + bg) - bg

    coords = flex.vec3_double(
        [
            (xs[ix] + 100.0, ys[iy] + 200.0, iz + 0.5)
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz)
        ]
    )
    intensities = flex.double(obs.ravel().astype(float) * gain)
    background_variances = flex.double(obs.size, bg * gain**2)
    gains = flex.double(obs.size, gain)
    return (
        coords,
        intensities,
        background_variances,
        flex.double(tof),
        (nx, ny, nz),
        model.sum() * gain,
        gains,
    )


def test_line_shape_matches_numerical_convolution():
    # rectangle (x) exponential (x) Gaussian on a 1 us grid
    dt = 1.0
    t = np.arange(-4000.0, 8000.0, dt)
    rect = ((t >= 0) & (t < W_ESS)).astype(float) / W_ESS
    expo = np.where(t >= 0, np.exp(-t / TAU) / TAU, 0.0)
    expo[t == 0] *= 0.5  # trapezoid rule at the onset
    gauss = np.exp(-0.5 * (t / SIGMA) ** 2) / (np.sqrt(2 * np.pi) * SIGMA)

    def conv(a, b):
        c = np.convolve(a, b) * dt
        return np.interp(t, 2 * t[0] + np.arange(len(c)) * dt, c)

    numerical = conv(conv(rect, expo), gauss)
    closed = line_shape(t)
    assert np.max(np.abs(closed - numerical)) / numerical.max() < 1e-3
    assert closed.sum() * dt == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "W,tau,sigma",
    [
        (W_ESS, 1e-9, SIGMA),
        (W_ESS, TAU, 1e-9),
        (1e-9, TAU, SIGMA),
        (W_ESS, 1e6, SIGMA),
        (W_ESS, TAU, 1e6),
        (W_ESS, 1e-9, 1e-9),
        (1e-9, 1e-9, 1e-9),
    ],
)
def test_line_shape_limits(W, tau, sigma):
    t = np.linspace(-1e5, 1e5, 2001)
    values = line_shape(t, 0.0, W, tau, sigma)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    for x in (-1e5, -50.0, 0.0, 50.0, 1e5):
        F = tof_emg_cdf(x, tau, sigma)
        assert np.isfinite(F) and 0.0 <= F <= 1.0


def test_short_pulse_limit_is_emg():
    # W -> 0 reduces to the exponentially modified Gaussian density
    t = np.linspace(-2000.0, 4000.0, 601)
    lam = 1.0 / TAU
    from math import erfc

    emg = np.array(
        [
            0.5
            * lam
            * np.exp(0.5 * lam * (lam * SIGMA**2 - 2 * x))
            * erfc((lam * SIGMA**2 - x) / (np.sqrt(2) * SIGMA))
            for x in t
        ]
    )
    # bin width of t is 10 us; the line shape is a bin mean, so compare
    # against the density at the bin centre to within the curvature error
    lp = line_shape(t, 0.0, 1e-2, TAU, SIGMA)
    assert np.max(np.abs(lp - emg)) / emg.max() < 2e-3


@pytest.mark.parametrize(
    "fit_kwargs,box_kwargs",
    [
        ({}, {}),
        ({"optimize_pulse_width": True}, {}),
        ({}, {"bin_us": 300.0, "nz": 24}),
        ({}, {"I_true": 2000.0}),
        # corrected units, as after the Lorentz correction on NMX data
        ({}, {"gain": 0.01}),
    ],
)
def test_round_trip(fit_kwargs, box_kwargs):
    rng = np.random.default_rng(1)
    weak = "I_true" in box_kwargs
    ratios, z_scores = [], []
    n = 12
    for _ in range(n):
        coords, intensities, bg_var, tof, (nx, ny, nz), I_box, gains = make_shoebox(
            rng, **box_kwargs
        )
        params = lp_params(**fit_kwargs)
        ok, I_prf, var_prf, profile = fit_profile_3d_lp(
            coords, intensities, bg_var, tof, nx, ny, nz, params, gains
        )
        # Weak reflections (I/sigma ~ 15 here) may be rejected by the trust
        # gates and fall back to summation; strong ones must all fit
        if not ok and weak:
            continue
        assert ok
        assert var_prf > 0
        assert len(profile) == nx * ny * nz
        ratios.append(I_prf / I_box)
        z_scores.append((I_prf - I_box) / np.sqrt(var_prf))
    assert len(ratios) >= (0.75 * n if weak else n)
    ratios = np.array(ratios)
    z_scores = np.array(z_scores)
    # Unbiased to within the scatter, and the propagated sigma is the right
    # size (assert on intensity, not on tau, which is degenerate with T0 and
    # sigma on the leading edge)
    assert abs(np.mean(z_scores)) < 3.0 / np.sqrt(len(z_scores)) + 0.2
    assert 0.5 < np.std(z_scores) < 1.6
    tol = 0.03 if weak else 0.01
    assert np.mean(ratios) == pytest.approx(1.0, abs=tol)


def test_refined_pulse_width_is_recovered():
    rng = np.random.default_rng(3)
    widths = []
    for _ in range(8):
        coords, intensities, bg_var, tof, (nx, ny, nz), _, gains = make_shoebox(
            rng, I_true=50000.0, W=2600.0
        )
        params = lp_params(optimize_pulse_width=True)
        ok, *_ = fit_profile_3d_lp(coords, intensities, bg_var, tof, nx, ny, nz, params)
        assert ok
        widths.append(params.pulse_width)
    assert np.mean(widths) == pytest.approx(2600.0, rel=0.02)


def test_long_pulse_beats_ikeda_carpenter_on_long_pulse_data():
    rng = np.random.default_rng(2)
    lp_ratios, ic_ratios = [], []
    for _ in range(6):
        coords, intensities, bg_var, tof, (nx, ny, nz), I_box, gains = make_shoebox(rng)
        ok_lp, I_lp, _, _ = fit_profile_3d_lp(
            coords, intensities, bg_var, tof, nx, ny, nz, lp_params()
        )
        ok_ic, I_ic = fit_profile_3d_ic(
            coords, intensities, bg_var, nx, ny, nz, ic_params()
        )
        assert ok_lp
        lp_ratios.append(I_lp / I_box)
        ic_ratios.append(I_ic / I_box if ok_ic else 0.0)
    lp_bias = abs(np.mean(lp_ratios) - 1.0)
    ic_bias = abs(np.mean(ic_ratios) - 1.0)
    assert lp_bias < 0.02
    assert ic_bias > 0.05


def true_library_shape(T0=1500.0, nx=11, ny=11, bin_us=100.0, scatter=(0.3, 0.3, 50.0)):
    """The shape make_shoebox draws with its defaults, as a library entry,
    expected at its true centre"""
    tof0 = 95000.0 + 0.5 * bin_us
    return {
        "W": W_ESS,
        "tau": TAU,
        "sigma": SIGMA,
        "sig_x": 1.5,
        "sig_y": 1.2,
        "sig_p": 0.2,
        "dxdt": 0.0,
        "dydt": 0.0,
        "expected": (
            0.5 * nx + 100.0,
            0.5 * ny + 200.0,
            tof0 + T0 + 0.5 * W_ESS + TAU,
        ),
        "scatter": scatter,
    }


@pytest.mark.parametrize("free_position_min_i_sigma", [10.0, 1e9])
def test_library_fit_weak_reflections(free_position_min_i_sigma):
    # Too weak for free fits to be accepted (summation I/sigma ~ 4), but
    # unbiased with a library shape, whether or not the position is freed
    rng = np.random.default_rng(4)
    ratios, z_scores, n_free = [], [], 0
    params = lp_params(free_position_min_i_sigma=free_position_min_i_sigma)
    for _ in range(20):
        coords, intensities, bg_var, tof, (nx, ny, nz), I_box, gains = make_shoebox(
            rng, I_true=500.0
        )
        ok, I_prf, var_prf, position_free, _ = fit_profile_3d_lp_library(
            coords,
            intensities,
            bg_var,
            tof,
            nx,
            ny,
            nz,
            params,
            true_library_shape(),
            gains,
        )
        assert ok
        assert var_prf > 0
        n_free += position_free
        ratios.append(I_prf / I_box)
        z_scores.append((I_prf - I_box) / np.sqrt(var_prf))
    if free_position_min_i_sigma > 1e6:
        assert n_free == 0
    assert abs(np.mean(z_scores)) < 3.0 / np.sqrt(len(z_scores)) + 0.2
    assert 0.5 < np.std(z_scores) < 1.6
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.1)


def test_library_fit_frees_position_of_strong_reflections():
    # A strong reflection 1 px and 200 us from where it is expected: its
    # position is refitted, and the intensity is unaffected
    rng = np.random.default_rng(5)
    shape = true_library_shape(scatter=(0.5, 0.5, 100.0))
    x, y, t = shape["expected"]
    shape["expected"] = (x + 1.0, y - 1.0, t + 200.0)
    for _ in range(5):
        coords, intensities, bg_var, tof, (nx, ny, nz), I_box, gains = make_shoebox(rng)
        ok, I_prf, var_prf, position_free, centre = fit_profile_3d_lp_library(
            coords, intensities, bg_var, tof, nx, ny, nz, lp_params(), shape, gains
        )
        assert ok and position_free
        assert centre[0] == pytest.approx(x, abs=0.3)
        assert centre[1] == pytest.approx(y, abs=0.3)
        assert centre[2] == pytest.approx(t, abs=60.0)
        assert I_prf / I_box == pytest.approx(1.0, abs=0.03)


def test_library_fit_corrected_units():
    rng = np.random.default_rng(6)
    ratios = []
    for _ in range(10):
        coords, intensities, bg_var, tof, (nx, ny, nz), I_box, gains = make_shoebox(
            rng, I_true=1000.0, gain=0.01
        )
        ok, I_prf, var_prf, _, _ = fit_profile_3d_lp_library(
            coords,
            intensities,
            bg_var,
            tof,
            nx,
            ny,
            nz,
            lp_params(),
            true_library_shape(),
            gains,
        )
        assert ok
        ratios.append(I_prf / I_box)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.08)
