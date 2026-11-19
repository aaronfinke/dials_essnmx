from __future__ import annotations

import math

import pytest

from dials.array_family import flex
from dials_algorithms_tof_integration_ext import line_shape_values

ALPHA, BETA, SIGMA = 0.08, 0.02, 6.0  # 1/us, 1/us, us


def pulse(tau):
    """
    The iBIX pulse, back-to-back exponentials (unit area) convolved with a
    Gaussian, at time tau after T_ph: an independent closed form.
    """
    a, b, s = ALPHA, BETA, SIGMA
    u = a / 2 * (a * s * s + 2 * tau)
    y = (a * s * s + tau) / (s * math.sqrt(2))
    v = b / 2 * (b * s * s - 2 * tau)
    z = (b * s * s - tau) / (s * math.sqrt(2))
    return (
        a
        * b
        / (2 * (a + b))
        * (math.exp(u) * math.erfc(y) + math.exp(v) * math.erfc(z))
    )


def test_gutmann_slices_hold_the_integral_of_the_pulse():
    """Each slice of the Gutmann line shape is the pulse integrated over it."""
    # Uneven slices, as logarithmic binning gives
    edges = [-200.0]
    while edges[-1] < 600.0:
        edges.append(edges[-1] + 4.0 + 0.02 * (edges[-1] + 200.0))
    centres = [0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]
    # slice edges of the line shape lie halfway between centres
    mid = [0.5 * (a + b) for a, b in zip(centres[:-1], centres[1:])]
    lo = [centres[0] - 0.5 * (centres[1] - centres[0])] + mid
    hi = mid + [centres[-1] + 0.5 * (centres[-1] - centres[-2])]

    t_ref = 0.0
    T_ph = t_ref - (1.0 / BETA - 1.0 / ALPHA)  # t_ref is the centroid
    binned = list(
        line_shape_values("gutmann", [ALPHA, BETA, SIGMA], flex.double(centres), t_ref)
    )
    for i in range(0, len(centres), 7):
        n = 400
        step = (hi[i] - lo[i]) / n
        integral = step * sum(pulse(lo[i] + (k + 0.5) * step - T_ph) for k in range(n))
        assert binned[i] == pytest.approx(integral, rel=1e-4, abs=1e-12)
    # and over a wide range the slices hold the whole pulse
    assert sum(binned) == pytest.approx(1.0, abs=1e-3)


def test_gutmann_centroid_is_at_the_reference_time():
    t = [-400.0 + 2.0 * k for k in range(800)]
    v = list(line_shape_values("gutmann", [ALPHA, BETA, SIGMA], flex.double(t), 37.0))
    centroid = sum(a * b for a, b in zip(t, v)) / sum(v)
    assert centroid == pytest.approx(37.0, abs=0.05)
    assert all(x >= 0.0 for x in v)
    assert all(math.isfinite(x) for x in v)
