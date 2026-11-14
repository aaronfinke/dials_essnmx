from __future__ import annotations

import json
from os.path import join

import numpy as np
import pytest

from dxtbx import flumpy
from dxtbx.model.experiment_list import ExperimentListFactory

from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.command_line.tof_integrate import (
    find_overlap_clusters,
    phil_scope,
    run_integrate,
)
from dials.util.phil import parse
from dials_algorithms_tof_integration_ext import (
    fit_overlap_cluster,
    tof_assign_overlap_ownership,
    tof_mask_neighbour_foreground,
)

VALID = int(MaskCode.Valid)
FOREGROUND = int(MaskCode.Foreground)
BACKGROUND = int(MaskCode.Background)
OVERLAPPED = int(MaskCode.Overlapped)


def make_table(bboxes, centroids, foregrounds=None):
    """
    A reflection table of allocated shoeboxes on one panel. foregrounds, if
    given, holds per reflection the (x0, x1, y0, y1, z0, z1) detector range to
    label Foreground; everything else is Background.
    """
    table = flex.reflection_table()
    table["bbox"] = flex.int6(bboxes)
    table["panel"] = flex.size_t(len(bboxes), 0)
    table["xyzcal.px"] = flex.vec3_double(centroids)
    shoeboxes = flex.shoebox(table["panel"], table["bbox"], allocate=True)
    for i, bbox in enumerate(bboxes):
        x0, x1, y0, y1, z0, z1 = bbox
        mask = np.full((z1 - z0, y1 - y0, x1 - x0), VALID | BACKGROUND, dtype=np.uint8)
        if foregrounds is not None:
            fx0, fx1, fy0, fy1, fz0, fz1 = foregrounds[i]
            mask[fz0 - z0 : fz1 - z0, fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0] = (
                VALID | FOREGROUND
            )
        shoebox = shoeboxes[i]
        shoebox.mask = flumpy.from_numpy(mask)
        shoeboxes[i] = shoebox
    table["shoebox"] = shoeboxes
    return table


def masks(table):
    return [flumpy.to_numpy(shoebox.mask) for shoebox in table["shoebox"]]


def test_ownership_gives_each_shared_voxel_to_one_reflection():
    # Boxes share x in [6, 10); the centroids are symmetric about x = 8
    bboxes = [(0, 10, 0, 10, 0, 20), (6, 16, 0, 10, 0, 20)]
    table = make_table(bboxes, [(5, 5, 10), (11, 5, 10)])

    has_overlap = tof_assign_overlap_ownership(table)
    assert list(has_overlap) == [True, True]

    mask_a, mask_b = masks(table)
    overlapped_a = (mask_a & OVERLAPPED) != 0
    overlapped_b = (mask_b & OVERLAPPED) != 0

    # Each shared voxel is Overlapped in exactly one of the two boxes, and
    # goes to the nearer centroid
    shared_a = overlapped_a[:, :, 6:10]
    shared_b = overlapped_b[:, :, 0:4]
    assert np.all(shared_a ^ shared_b)
    assert not shared_a[:, :, :2].any() and shared_a[:, :, 2:].all()

    # Nothing outside the shared range is touched
    assert not overlapped_a[:, :, :6].any()
    assert not overlapped_b[:, :, 4:].any()


def test_ownership_without_overlap():
    table = make_table(
        [(0, 10, 0, 10, 0, 20), (10, 20, 0, 10, 0, 20)], [(5, 5, 10), (15, 5, 10)]
    )
    assert list(tof_assign_overlap_ownership(table)) == [False, False]
    assert not any((mask & OVERLAPPED).any() for mask in masks(table))
    first, second = tof_mask_neighbour_foreground(table)
    assert len(first) == 0 and len(second) == 0


def test_padding_overlap_leaves_both_foregrounds_clean():
    # The boxes overlap, but each foreground stays inside its own voxels and
    # clear of the other box
    bboxes = [(0, 10, 0, 10, 0, 20), (6, 16, 0, 10, 0, 20)]
    foregrounds = [(3, 6, 3, 7, 8, 12), (10, 13, 3, 7, 8, 12)]
    table = make_table(bboxes, [(4.5, 5, 10), (11.5, 5, 10)], foregrounds)

    tof_assign_overlap_ownership(table)
    first, second = tof_mask_neighbour_foreground(table)
    assert len(first) == 0

    for mask in masks(table):
        foreground = (mask & FOREGROUND) != 0
        assert foreground.any()
        assert not (mask[foreground] & OVERLAPPED).any()


def test_neighbour_foreground_is_removed_from_background():
    # The foreground of the second reflection sits in voxels it owns, but
    # inside the first reflection's box
    bboxes = [(0, 10, 0, 10, 0, 20), (6, 16, 0, 10, 0, 20)]
    foregrounds = [(2, 5, 3, 7, 8, 12), (8, 12, 3, 7, 8, 12)]
    table = make_table(bboxes, [(3.5, 5, 10), (10, 5, 10)], foregrounds)

    tof_assign_overlap_ownership(table)
    first, second = tof_mask_neighbour_foreground(table)
    assert len(first) == 0  # neither foreground reaches the other's voxels

    mask_a, mask_b = masks(table)
    # In the first box those voxels are background, and now flagged
    region = mask_a[8:12, 3:7, 8:10]
    assert np.all((region & BACKGROUND) != 0)
    assert np.all((region & OVERLAPPED) != 0)
    # Neither foreground is affected
    for mask in (mask_a, mask_b):
        foreground = (mask & FOREGROUND) != 0
        assert not (mask[foreground] & OVERLAPPED).any()


def test_overlapping_foregrounds_are_coupled():
    bboxes = [
        (0, 10, 0, 10, 0, 20),
        (6, 16, 0, 10, 0, 20),
        (40, 50, 0, 10, 0, 20),
    ]
    foregrounds = [(4, 9, 3, 7, 8, 12), (7, 12, 3, 7, 8, 12), (43, 47, 3, 7, 8, 12)]
    table = make_table(bboxes, [(6.5, 5, 10), (9.5, 5, 10), (45, 5, 10)], foregrounds)

    tof_assign_overlap_ownership(table)
    first, second = tof_mask_neighbour_foreground(table)
    assert list(first) == [0] and list(second) == [1]

    mask_a, mask_b, mask_c = masks(table)
    for mask in (mask_a, mask_b):
        assert ((mask & (FOREGROUND | OVERLAPPED)) == (FOREGROUND | OVERLAPPED)).any()
    assert not (mask_c & OVERLAPPED).any()

    assert list(find_overlap_clusters(3, (first, second))) == [0, 0, -1]


def test_find_overlap_clusters_joins_chains():
    pairs = (flex.size_t([0, 1, 5]), flex.size_t([1, 2, 6]))
    assert list(find_overlap_clusters(8, pairs)) == [0, 0, 0, -1, -1, 1, 1, -1]


def two_peaks(separation):
    x = np.arange(60, dtype=float)
    centre = 30.0
    profiles = np.stack(
        [
            np.exp(-0.5 * ((x - centre + 0.5 * separation) / 3.0) ** 2),
            np.exp(-0.5 * ((x - centre - 0.5 * separation) / 3.0) ** 2),
        ],
        axis=1,
    )
    return profiles / profiles.sum(axis=0)


def fit(profiles, data, variance):
    return [
        np.array(a)
        for a in fit_overlap_cluster(
            flumpy.from_numpy(np.ascontiguousarray(profiles)),
            flex.double(data),
            flex.double(variance),
        )
    ]


def test_fit_overlap_cluster_recovers_known_intensities():
    profiles = two_peaks(separation=5.0)
    truth = np.array([1000.0, 300.0])
    data = profiles @ truth
    variance = np.ones(len(data))

    value, value_variance = fit(profiles, data, variance)
    assert value == pytest.approx(truth)
    assert np.all(value_variance > 0)


def test_fit_overlap_cluster_variance_is_calibrated():
    # Over many noisy realisations the scatter of the estimate must match the
    # variance it reports
    rng = np.random.default_rng(0)
    profiles = two_peaks(separation=4.0)
    truth = np.array([1000.0, 300.0])
    variance = np.full(profiles.shape[0], 4.0)
    estimates = []
    for _ in range(400):
        data = profiles @ truth + rng.normal(0.0, 2.0, profiles.shape[0])
        value, value_variance = fit(profiles, data, variance)
        estimates.append(value)
    estimates = np.array(estimates)
    assert estimates.mean(axis=0) == pytest.approx(truth, rel=0.02)
    assert estimates.var(axis=0) == pytest.approx(value_variance, rel=0.25)


def test_fit_overlap_cluster_degenerate_fails():
    # Two reflections with the same profile: only their sum is measured, so
    # the fit refuses
    profile = two_peaks(separation=0.0)
    data = profile @ np.array([700.0, 300.0])
    variance = np.ones(len(data))
    _, value_variance = fit(profile, data, variance)
    assert np.all(value_variance < 0)


@pytest.fixture(scope="module")
def sxd_nacl(dials_data, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tof_overlaps")
    image_file = join(dials_data("isis_sxd_example_data"), "sxd_nacl_run.nxs")
    with open(join(dials_data("isis_sxd_nacl_processed"), "refined.expt"), "rb") as g:
        expt_json = json.load(g)
    expt_json["imageset"][0]["template"] = image_file
    expt_path = join(tmp_path, "refined.expt")
    with open(expt_path, "w") as g:
        json.dump(expt_json, g)
    refl_path = join(dials_data("isis_sxd_nacl_processed"), "refined.refl")
    return expt_path, refl_path


def integrate(sxd_nacl, *args):
    expt_path, refl_path = sxd_nacl
    params = phil_scope.fetch(parse("\n".join(args))).extract()
    experiments = ExperimentListFactory.from_json_file(expt_path)
    reflections = flex.reflection_table.from_msgpack_file(refl_path)
    return experiments, run_integrate(params, experiments, reflections)


def keyed(reflections):
    return {
        (h, p, round(xyz[2])): i
        for i, (h, p, xyz) in enumerate(
            zip(
                reflections["miller_index"],
                reflections["panel"],
                reflections["xyzcal.px"],
            )
        )
    }


def test_mask_mode_keeps_overlapped_reflections(sxd_nacl):
    # Boxes padded until some of them overlap
    settings = ["bbox_xy_padding=6", "mp.nproc=1"]
    _, rejected = integrate(sxd_nacl, "overlaps.mode=reject", *settings)
    _, masked = integrate(sxd_nacl, "overlaps.mode=mask", *settings)

    assert len(masked) > len(rejected)
    flags = masked.flags
    overlapped_fg = masked.get_flags(flags.overlapped_fg)
    integrated = masked.get_flags(flags.integrated_sum)
    # An overlapped foreground is never summed; everything else here is
    assert (overlapped_fg & integrated).count(True) == 0
    assert integrated.count(True) > rejected.get_flags(flags.integrated_sum).count(True)

    # A reflection that never overlapped is integrated exactly as before
    index = keyed(masked)
    for key, i in keyed(rejected).items():
        j = index[key]
        assert masked["intensity.sum.value"][j] == pytest.approx(
            rejected["intensity.sum.value"][i]
        )


def test_deconvolve_mode(sxd_nacl):
    with pytest.raises(ValueError):
        integrate(sxd_nacl, "overlaps.mode=deconvolve", "method=summation")

    # Boxes and foregrounds large enough for some foregrounds to overlap
    settings = [
        "overlaps.mode=deconvolve",
        "method=profile_1d_profilelib",
        "bbox_xy_padding=10",
        "bbox_tof_padding=4",
        "ellipse_mask.scale=2",
        "mp.nproc=1",
    ]
    _, reflections = integrate(sxd_nacl, *settings)

    flags = reflections.flags
    overlapped_fg = reflections.get_flags(flags.overlapped_fg)
    fitted = reflections["overlap.fit.variance"] > 0
    assert overlapped_fg.count(True) > 0
    assert (overlapped_fg & fitted).count(True) > 0
    assert (fitted & (reflections["overlap.cluster_id"] < 0)).count(True) == 0

    # A deconvolved reflection carries the joint fit as its profile intensity
    deconvolved = overlapped_fg & fitted
    assert (deconvolved & ~reflections.get_flags(flags.integrated_prf)).count(True) == 0
    sel = reflections.select(deconvolved)
    assert list(sel["intensity.prf.value"]) == pytest.approx(
        list(sel["overlap.fit.value"])
    )
    assert list(sel["intensity.prf.variance"]) == pytest.approx(
        list(sel["overlap.fit.variance"])
    )


@pytest.mark.parametrize("mode", ["mask", "deconvolve"])
def test_joint_estimator(sxd_nacl, mode):
    with pytest.raises(ValueError):
        integrate(sxd_nacl, "overlaps.estimator=joint", "method=summation")

    settings = [
        f"overlaps.mode={mode}",
        "overlaps.estimator=joint",
        "method=profile_1d_profilelib",
        "bbox_xy_padding=10",
        "bbox_tof_padding=4",
        "ellipse_mask.scale=2",
        "mp.nproc=1",
    ]
    _, reflections = integrate(sxd_nacl, *settings)
    flags = reflections.flags
    prf = reflections.get_flags(flags.integrated_prf)
    fitted = reflections["overlap.fit.variance"] > 0
    assert prf.count(True) > 0

    # Every profile-fitted intensity is the 3D fit
    assert (prf & ~fitted).count(True) == 0
    sel = reflections.select(prf)
    assert list(sel["intensity.prf.value"]) == pytest.approx(
        list(sel["overlap.fit.value"])
    )
    assert list(sel["intensity.prf.variance"]) == pytest.approx(
        list(sel["overlap.fit.variance"])
    )

    # The 1D fit is kept beside it, and agrees on strong reflections
    one_d = reflections["intensity.prf_1d.variance"] > 0
    assert one_d.count(True) > 0
    both = reflections.select(prf & one_d)
    strong = both["intensity.prf_1d.value"] > 10 * flex.sqrt(
        both["intensity.prf_1d.variance"]
    )
    assert strong.count(True) > 10
    assert (
        flex.linear_correlation(
            both["intensity.prf.value"].select(strong),
            both["intensity.prf_1d.value"].select(strong),
        ).coefficient()
        > 0.9
    )

    # Overlapped foregrounds are fitted only when deconvolving
    overlapped_fg = reflections.get_flags(flags.overlapped_fg)
    assert overlapped_fg.count(True) > 0
    n_overlapped_fitted = (overlapped_fg & prf).count(True)
    if mode == "mask":
        assert n_overlapped_fitted == 0
    else:
        assert n_overlapped_fitted > 0
