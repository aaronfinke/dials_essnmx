# LIBTBX_SET_DISPATCHER_NAME dials.tof_integrate
from __future__ import annotations

import logging
import multiprocessing
import os
import re
import sys
import warnings
from copy import deepcopy
from math import ceil, floor
from typing import Dict, Optional, Tuple

import numpy as np

import cctbx.array_family.flex
import libtbx
from dxtbx import flumpy
from dxtbx.model import Experiment, ExperimentList, ImageSequence

import dials.util.log
from dials.algorithms.integration.integrator import (
    generate_phil_scope as integrator_phil_scope,
)
from dials.algorithms.integration.report import IntegrationReport
from dials.algorithms.shoebox import MaskCode
from dials.algorithms.spot_prediction import TOFReflectionPredictor
from dials.array_family import flex
from dials.command_line.integrate import process_reference
from dials.extensions.simple_background_ext import SimpleBackgroundExt
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_algorithms_tof_integration_ext import (
    TOFProfile1DIBIXParams,
    TOFProfile1DICParams,
    TOFProfile1DProfilelibParams,
    TOFProfile3DGutmannParams,
    TOFProfile3DIBIXParams,
    TOFProfile3DICParams,
    integrate_reflection_table,
    tof_assign_overlap_ownership,
    tof_calculate_ellipse_shoebox_mask,
    tof_calculate_seed_skewness_shoebox_mask,
    tof_deconvolve_overlaps,
    tof_mask_neighbour_foreground,
)
from dials_tof_scaling_ext import (
    TOFAbsorptionParams,
    TOFIncidentSpectrumParams,
    tof_extract_shoeboxes_to_reflection_table,
)

logger = logging.getLogger("dials.command_line.tof_integrate")

phil_scope = parse(
    """
output {
experiments = 'integrated.expt'
    .type = str
    .help = "The experiments output filename"
reflections = 'integrated.refl'
    .type = str
    .help = "The integrated output filename"
phil = 'dials.simple_integrate.phil'
    .type = str
    .help = "The output phil file"
log = 'tof_integrate.log'
    .type = str
    .help = "The log filename"
}
integration_type = *observed calculated
    .type = choice
    .help = "observed integrates only reflections observed during spot finding"
            "calculated integrates all reflections out to calculated.dmin"

background_model = constant2d constant3d linear2d *linear3d
    .type = choice
    .help = Model used to calculate the background of each reflection

calculated{
    dmin = none
    .type = float (value_min=0.5)
    .help = "The resolution spots are integrated to when using integration_type.calculated"

}
method = *summation profile_1d_ibix profile_1d_profilelib profile_1d_ic profile_3d_gutmann profile_3d_ic profile_3d_ibix
    .type = choice
    .help = "Integration method: "
            "summation: shoebox summation"
            "profile_1d_ibix: https://doi.org/10.1038/srep36628 "
            "profile_1d_profilelib: profile_1d_ibix's model, with the shape taken "
            "from a library measured on strong reflections, after Mantid's "
            "IntegratePeaksProfileFitting "
            "profile_1d_ic: Ikeda-Carpenter model, "
            "https://doi.org/10.1016/0168-9002(85)90033-6"
            "profile_3d_gutmann: https://doi.org/10.1016/j.nima.2016.12.026"
            "profile_3d_ic: separable Ikeda-Carpenter x bivariate Gaussian model, "
            "https://doi.org/10.1016/0168-9002(85)90033-6"
            "profile_3d_ibix: separable iBIX x bivariate Gaussian model, "
            "https://doi.org/10.1038/srep36628"

mask = *ellipse seed_skewness
    .type = choice
    .help = "Foreground/background mask method: "
            "seed_skewness: https://doi.org/10.1107/S0021889803021939"
ellipse_mask{
    scale = 1.0
    .type = float (value_min=0.5)
    .help = "Number of standard deviations to use when generating the ellipse mask"
}


corrections{
    incident_run = None
        .type = str
        .help = "Path to incident run to normalize intensities (e.g. Vanadium)."
    empty_run = None
        .type = str
        .help = "Path to empty run to correct empty counts."
    lorentz = False
        .type = bool
        .help = "Apply the Lorentz correction to target spectrum."
    lorentz_wavelength_power = 4.0
        .type = float(value_min=0)
        .help = "Exponent p in the Lorentz factor sin^2(theta) / lambda^p. The"
                "default 4 is the full factor. 0 applies sin^2(theta) alone,"
                "leaving the wavelength part to be absorbed by the scaling"
                "program's normalisation curve; applied per ToF slice it"
                "reweights the inside of a peak and distorts the profile being"
                "fitted."
    absorption{
        incident_spectrum{
            sample_number_density = 0.0722
                .type = float
                .help = "Sample number density for incident run."
                        "Default is Vanadium used at SXD"
            sample_radius = 0.03
                .type = float
                .help = "Sample radius incident run."
                        "Default is Vanadium used at SXD"
            scattering_x_section = 5.158
                .type = float
                .help = "Sample scattering cross section used for incident run."
                        "Default is Vanadium used at SXD"
            absorption_x_section = 4.4883
                .type = float
                .help = "Sample absorption cross section for incident run."
                        "Default is Vanadium used at SXD"
        }
        target_spectrum{
            sample_number_density = None
                .type = float
                .help = "Sample number density for target run."
            sample_radius = None
                .type = float
                .help = "Sample radius target run."
            scattering_x_section = None
                .type = float
                .help = "Sample scattering cross section used for target run."
            absorption_x_section = None
                .type = float
                .help = "Sample absorption cross section for target run."
        }
    }
}
profile_1d_ibix{
    init_alpha = 0.03
        .type = float
        .help = "Initial alpha value before optimization"
    init_beta = 0.03
        .type = float
        .help = "Initial beta value before optimization"
    min_alpha = 0.02
        .type = float
        .help = "Min alpha value for optimization"
    max_alpha = 1.0
        .type = float
        .help = "Max alpha value for optimization"
    min_beta = 0.0
        .type = float
        .help = "Min beta value for optimization"
    max_beta = 1.0
        .type = float
        .help = "Max beta value for optimization"
    min_A = 1.0
        .type = float
        .help = "Min A value for optimization"
    max_A = 1e4
        .type = float
        .help = "Max A value for optimization"
    n_restarts = 100
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"

}
profile_1d_profilelib
    .help = "Profile fitting against a library of peak shapes measured on"
            "strong reflections, after Mantid's IntegratePeaksProfileFitting."
            "The shape model is the same back-to-back exponential"
            "profile_1d_ibix fits; the difference is that a weak reflection"
            "takes the shape of the nearest strong one and fits only its"
            "amplitude and position, rather than being asked to determine a"
            "shape its own counts cannot support."
{
    init_alpha = 0.03
        .type = float
        .help = "Initial alpha value before optimization. Not critical: the"
                "bounds below decide the outcome, and 0.03 against 0.005 moves"
                "nothing measurable on MANDI."
    init_beta = 0.03
        .type = float
        .help = "Initial beta value before optimization"
    min_alpha = 1e-5
        .type = float
        .help = "Min alpha value for optimization. Wide enough for a"
                "short-pulse source: MANDI peaks fit alpha near 0.005, so the"
                "0.02 that suits a long-pulse instrument is a floor the data"
                "cannot reach, and a parameter pinned to it gives a rising edge"
                "far too sharp to match the peak."
    max_alpha = 10.0
        .type = float
        .help = "Max alpha value for optimization"
    min_beta = 1e-5
        .type = float
        .help = "Min beta value for optimization. Not zero: beta = 0 makes the"
                "profile identically zero."
    max_beta = 10.0
        .type = float
        .help = "Max beta value for optimization"
    min_A = 1.0
        .type = float
        .help = "Min A value for optimization"
    max_A = 1e4
        .type = float
        .help = "Max A value for optimization"
    n_restarts = 100
        .type = int(value_min=0)
        .help = "If a shape fit fails, number of additional attempts with"
                "perturbed params"
    library{
        min_i_sigma = 5.0
            .type = float(value_min=0)
            .help = "Summation I/sigma a reflection must reach for its fitted"
                    "shape to enter the library."
        min_correlation = 0.5
            .type = float(value_min=0, value_max=1)
            .help = "Correlation a free fit must reach for its shape to enter"
                    "the library. Looser than the threshold for accepting an"
                    "intensity, because a shape only has to be the right shape."
    }
    peak_tolerance = 300.0
        .type = float(value_min=0)
        .help = "How far in us a shape fit's peak may sit from the tallest"
                "channel of the data. A time rather than a number of bins, so"
                "the tolerance does not shrink as the slices get finer. 300 us"
                "matches the fixed three bins profile_1d_ibix uses at 101 us."
    peak_height_smoothing = 60.0
        .type = float(value_min=0)
        .help = "Time in us to smooth the data over before measuring its peak"
                "height. The tallest single channel is the maximum of as many"
                "samples as there are bins, so on finely sliced data it sits"
                "above the curve it is drawn from and no correct fit can reach"
                "it. Set to 0 to compare against the raw maximum."
    peak_height_fraction = 0.1
        .type = float(value_min=0)
        .help = "Allowed relative difference between a shape fit's peak height"
                "and the height of the data."
}
profile_1d_ic{
    init_A = 1.0
        .type = float
        .help = "Initial fast-neutron decay constant before optimization"
    min_A = 0.1
        .type = float
        .help = "Min fast-neutron decay constant for optimization"
    max_A = 20.0
        .type = float
        .help = "Max fast-neutron decay constant for optimization"
    init_B = 0.05
        .type = float
        .help = "Initial slow-neutron decay constant before optimization"
    min_B = 1e-4
        .type = float
        .help = "Min slow-neutron decay constant for optimization"
    max_B = 5.0
        .type = float
        .help = "Max slow-neutron decay constant for optimization"
    init_R = 0.5
        .type = float
        .help = "Initial fast/slow neutron ratio before optimization"
    min_R = 0.0
        .type = float
        .help = "Min fast/slow neutron ratio for optimization"
    max_R = 1.0
        .type = float
        .help = "Max fast/slow neutron ratio for optimization"
    hat_width = 1.0
        .type = float
        .help = "Initial half-width of the rectangular convolution kernel (ToF bins)"
    kconv = 0.1
        .type = float
        .help = "Initial Gaussian convolution kernel decay rate (per ToF bin^2)"
    n_restarts = 100
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"
}
profile_3d_gutmann{
    init_alpha = 1.0
        .type = float
        .help = "Initial alpha value before optimization"
    init_beta = 0.1
        .type = float
        .help = "Initial beta value before optimization"
    min_alpha = 0.5
        .type = float
        .help = "Min alpha value for optimization"
    max_alpha = 20.
        .type = float
        .help = "Max alpha value for optimization"
    min_beta = 1e-7
        .type = float
        .help = "Min beta value for optimization"
    max_beta = 5.0
        .type = float
        .help = "Max beta value for optimization"
    n_restarts = 100
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"
    gradient_method = *forward_difference central_difference
        .type = choice
        .help = "Method used to calculate gradients"
}
profile_3d_ic{
    init_A = 1.0
        .type = float
        .help = "Initial fast-neutron decay constant before optimization"
    min_A = 0.1
        .type = float
        .help = "Min fast-neutron decay constant for optimization"
    max_A = 20.0
        .type = float
        .help = "Max fast-neutron decay constant for optimization"
    init_B = 0.05
        .type = float
        .help = "Initial slow-neutron decay constant before optimization"
    min_B = 1e-4
        .type = float
        .help = "Min slow-neutron decay constant for optimization"
    max_B = 5.0
        .type = float
        .help = "Max slow-neutron decay constant for optimization"
    init_R = 0.5
        .type = float
        .help = "Initial fast/slow neutron ratio before optimization"
    min_R = 0.0
        .type = float
        .help = "Min fast/slow neutron ratio for optimization"
    max_R = 1.0
        .type = float
        .help = "Max fast/slow neutron ratio for optimization"
    min_sig_x = 0.05
        .type = float
        .help = "Min bivariate Gaussian sigma X for optimization"
    max_sig_x = 20.0
        .type = float
        .help = "Max bivariate Gaussian sigma X for optimization"
    min_sig_y = 0.05
        .type = float
        .help = "Min bivariate Gaussian sigma Y for optimization"
    max_sig_y = 20.0
        .type = float
        .help = "Max bivariate Gaussian sigma Y for optimization"
    init_sig_p = 0.0
        .type = float
        .help = "Initial bivariate Gaussian spatial correlation before optimization"
    min_sig_p = -0.99
        .type = float
        .help = "Min bivariate Gaussian spatial correlation for optimization"
    max_sig_p = 0.99
        .type = float
        .help = "Max bivariate Gaussian spatial correlation for optimization"
    hat_width = 1.0
        .type = float
        .help = "Initial half-width of the rectangular convolution kernel (ToF bins)"
    kconv = 0.1
        .type = float
        .help = "Initial Gaussian convolution kernel decay rate (per ToF bin^2)"
    n_restarts = 100
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"
    max_drift_factor = 1.5
        .type = float
        .help = "Scales the per-reflection dXdt/dYdt bounds relative to the "
                "shoebox's spatial extent divided by its ToF extent"
}
profile_3d_ibix{
    init_alpha = 0.03
        .type = float
        .help = "Initial alpha value before optimization"
    min_alpha = 0.02
        .type = float
        .help = "Min alpha value for optimization"
    max_alpha = 1.0
        .type = float
        .help = "Max alpha value for optimization"
    init_beta = 0.03
        .type = float
        .help = "Initial beta value before optimization"
    min_beta = 0.0
        .type = float
        .help = "Min beta value for optimization"
    max_beta = 1.0
        .type = float
        .help = "Max beta value for optimization"
    min_sigma = 0.1
        .type = float
        .help = "Min instrument-resolution Gaussian width for optimization"
    max_sigma = 20.0
        .type = float
        .help = "Max instrument-resolution Gaussian width for optimization"
    min_sig_x = 0.05
        .type = float
        .help = "Min bivariate Gaussian sigma X for optimization"
    max_sig_x = 20.0
        .type = float
        .help = "Max bivariate Gaussian sigma X for optimization"
    min_sig_y = 0.05
        .type = float
        .help = "Min bivariate Gaussian sigma Y for optimization"
    max_sig_y = 20.0
        .type = float
        .help = "Max bivariate Gaussian sigma Y for optimization"
    init_sig_p = 0.0
        .type = float
        .help = "Initial bivariate Gaussian spatial correlation before optimization"
    min_sig_p = -0.99
        .type = float
        .help = "Min bivariate Gaussian spatial correlation for optimization"
    max_sig_p = 0.99
        .type = float
        .help = "Max bivariate Gaussian spatial correlation for optimization"
    n_restarts = 100
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"
    max_drift_factor = 1.5
        .type = float
        .help = "Scales the per-reflection dXdt/dYdt bounds relative to the "
                "shoebox's spatial extent divided by its ToF extent"
}

mp{
    nproc = auto
        .type = int(value_min=1)
        .help = "Number of processors to use during parallelized steps."
        "If set to auto DIALS will choose automatically."
}
bbox_tof_padding = 2
    .type = int
    .help = "Additional ToF frames added to calculated bounding boxes"
bbox_xy_padding = 1
    .type = int
    .help = "Additional pixels added to calculated bounding boxes"
keep_shoeboxes = False
    .type = bool
    .help = "Retain shoeboxes in output reflection table"

wavelength_range = None
    .type = floats(size=2)
    .help = "Reflections outside of this wavelength range are not considered (A)"

tof_range = None
    .type = floats(size=2)
    .help = "Reflections outside of this time-of-flight range are not considered (usec)"

overlaps
    .help = "What to do with reflections whose bounding boxes overlap."
{
    mode = *reject mask deconvolve
        .type = choice
        .help = "reject: remove every reflection whose bounding box overlaps"
                "another. "
                "mask: keep them. Each shared voxel is given to the reflection"
                "whose predicted centroid is nearer, and a voxel in one"
                "reflection's foreground is excluded from every other"
                "reflection's background and profile. Only a reflection whose"
                "own foreground reaches a neighbour's voxels is lost: it is"
                "flagged overlapped_fg and not integrated. "
                "deconvolve: as mask, then fit each cluster of overlapped"
                "reflections jointly in 3D, with shapes taken from the profile"
                "library and, if overlaps.reference is given, a prior on each"
                "intensity. Requires method=profile_1d_profilelib. Peak"
                "positions are held at their predictions, corrected by the"
                "offsets the library measured nearby, and a peak's position on"
                "the detector is taken not to drift with ToF."
    reference = None
        .type = path
        .help = "Merged intensities from data integrated without overlaps"
                "(a merged .mtz), used as a prior on the intensity of each"
                "overlapped reflection. It must be indexed consistently with"
                "the input experiments. Without it the overlaps are separated"
                "by shape alone."
    prior_sigma_scale = 1.0
        .type = float(value_min=0.01)
        .help = "Factor applied to the width of the intensity prior. Raise it"
                "to weaken the prior if the deconvolved intensities are being"
                "pulled towards the reference."
    min_reference_cc = 0.5
        .type = float(value_min=-1, value_max=1)
        .help = "Correlation the reference must reach with an experiment's"
                "own unoverlapped intensities, once scaled to them, for it to"
                "be used as a prior there. A reference indexed differently"
                "from the experiments falls below this."
    n_wavelength_bins = 20
        .type = int(value_min=1)
        .help = "Number of wavelength bins in the scale between the reference"
                "and an experiment's own unoverlapped intensities."
    estimator = *separate joint
        .type = choice
        .help = "separate: reflections fitted on their own keep the 1D profile"
                "fit, and only the deconvolved ones come from the 3D joint fit,"
                "divided by the median ratio of the two fits on strong"
                "unoverlapped reflections. "
                "joint: every reflection is fitted with the 3D model (one that"
                "overlaps nothing as a cluster of its own, without a prior), so"
                "that all profile-fitted intensities come from one estimator."
                "The 1D profile fit is kept as intensity.prf_1d. Works with"
                "every overlaps.mode; requires method=profile_1d_profilelib."
}

"""
)

"""
Usage:
$ dials.tof_integrate.py refined.expt refined.refl
$ dials.tof_integrate refined.expt refined.refl corrections.incident_run=vanadium_run.nxs corrections.empty_run=empty_run.nxs corrections.lorentz=True bbox_tof_padding=15 bbox_xy_padding=5 mask=ellipse method=profile_3d_gutmann mp.nproc=16 background_model=linear3d
"""

phil_scope.adopt_scope(integrator_phil_scope())

# Temporary parameter name mapping before deprecating
_DEPRECATED_PARAMETER_NAMES = {
    "profile1d": "profile_1d_ibix",
    "profile3d": "profile_3d_gutmann",
}


def _translate_deprecated_args(args):
    """
    Rewrite uses of deprecated parameter/choice names to their replacements,
    issuing a warning for each one found.
    """
    translated_args = []
    for arg in args:
        for old_name, new_name in _DEPRECATED_PARAMETER_NAMES.items():
            pattern = rf"(?<![\w.]){re.escape(old_name)}(?!\w)"
            new_arg = re.sub(pattern, new_name, arg)
            if new_arg != arg:
                warnings.warn(
                    f"'{old_name}' is deprecated and will be removed in a "
                    f"future release, please use '{new_name}' instead.",
                    UserWarning,
                    stacklevel=2,
                )
                arg = new_arg
        translated_args.append(arg)
    return translated_args


def get_corrections_data(
    experiments: ExperimentList, params: libtbx.phil.scope_extract
) -> Dict:
    """
    Extracts all the corrections in params related to incident
    and absorption corrections
    """

    corrections = {}

    experiment_cls = experiments[0].imageset.get_format_class()
    incident_proton_charge = None
    empty_proton_charge = None

    if applying_incident_and_empty_runs(params):
        incident_fmt_class = experiment_cls.get_instance(
            params.corrections.incident_run
        )
        incident_data = experiment_cls(params.corrections.incident_run).get_imageset(
            params.corrections.incident_run
        )
        corrections["incident_data"] = incident_data

        empty_data = experiment_cls(params.corrections.empty_run).get_imageset(
            params.corrections.empty_run
        )
        corrections["empty_data"] = empty_data
        empty_fmt_class = experiment_cls.get_instance(params.corrections.empty_run)
        incident_proton_charge = incident_fmt_class.get_proton_charge()
        empty_proton_charge = empty_fmt_class.get_proton_charge()

        corrections["incident_proton_charge"] = incident_proton_charge
        corrections["empty_proton_charge"] = empty_proton_charge

        if applying_spherical_absorption_correction(params):
            corrections["absorption_params"] = TOFAbsorptionParams(
                params.corrections.absorption.target_spectrum.sample_radius,
                params.corrections.absorption.target_spectrum.scattering_x_section,
                params.corrections.absorption.target_spectrum.absorption_x_section,
                params.corrections.absorption.target_spectrum.sample_number_density,
                params.corrections.absorption.incident_spectrum.sample_radius,
                params.corrections.absorption.incident_spectrum.scattering_x_section,
                params.corrections.absorption.incident_spectrum.absorption_x_section,
                params.corrections.absorption.incident_spectrum.sample_number_density,
            )

    return corrections


def calculate_shoebox_masks(
    experiment: Experiment,
    reflections: flex.reflection_table,
    params: libtbx.phil.scope_extract,
) -> flex.reflection_table:
    nproc = params.mp.nproc
    method = params.mask
    if method == "seed_skewness":
        logger.info("    Calculating seed skewness foreground/background mask")
        tof_calculate_seed_skewness_shoebox_mask(
            reflections, experiment, 1e-7, 10, nproc
        )
    else:
        logger.info("    Calculating ellipse foreground/background mask")
        scale = params.ellipse_mask.scale
        tof_calculate_ellipse_shoebox_mask(reflections, experiment, nproc, scale)

    return reflections


def get_spectrum_correction_params(**kwargs: Dict) -> Tuple:
    """
    The incident spectrum and absorption correction parameters for one
    experiment, either of which is None if the correction is not applied.
    kwargs are the output of get_corrections_data, with expt_proton_charge.
    """
    incident_params = None
    absorption_params = None
    if "incident_data" in kwargs:
        incident_params = TOFIncidentSpectrumParams(
            kwargs["incident_data"],
            kwargs["empty_data"],
            kwargs["expt_proton_charge"],
            kwargs["incident_proton_charge"],
            kwargs["empty_proton_charge"],
        )
        if "absorption_params" in kwargs:
            absorption_params = kwargs["absorption_params"]
    return incident_params, absorption_params


def integrate_reflection_table_for_experiment(
    expt: Experiment,
    expt_reflections: flex.reflection_table,
    expt_data: ImageSequence,
    params: libtbx.phil.scope_extract,
    **kwargs: Dict,
) -> Tuple[flex.reflection_table, object]:
    """
    Integrates expt_reflections in place. Returns them with the profile fitter
    used, which holds any profile library it built, or None for summation.
    """
    apply_lorentz = params.corrections.lorentz
    profile_1d_ibix_params = None
    profile_1d_profilelib_params = None
    profile_1d_ic_params = None
    profile_3d_gutmann_params = None
    profile_3d_ic_params = None
    profile_3d_ibix_params = None

    logger.info(f"    Integrating using {params.method}")

    show_profile_failures = logger.getEffectiveLevel() == logging.DEBUG
    if params.method == "profile_1d_ibix":
        alpha = params.profile_1d_ibix.init_alpha
        beta = params.profile_1d_ibix.init_beta
        min_A = params.profile_1d_ibix.min_A
        max_A = params.profile_1d_ibix.max_A
        if min_A > max_A:
            raise ValueError(
                f"profile_1d_ibix.min_A ({min_A}) is larger than "
                f"profile_1d_ibix.max_A ({max_A})"
            )
        min_alpha = params.profile_1d_ibix.min_alpha
        max_alpha = params.profile_1d_ibix.max_alpha
        min_beta = params.profile_1d_ibix.min_beta
        max_beta = params.profile_1d_ibix.max_beta
        n_restarts = params.profile_1d_ibix.n_restarts
        profile_1d_ibix_params = TOFProfile1DIBIXParams(
            min_A,
            max_A,
            alpha,
            min_alpha,
            max_alpha,
            beta,
            min_beta,
            max_beta,
            n_restarts,
            True,
            show_profile_failures,
        )
    elif params.method == "profile_1d_profilelib":
        p = params.profile_1d_profilelib
        if p.min_alpha > p.max_alpha:
            raise ValueError("profile_1d_profilelib.min_alpha > max_alpha")
        if p.min_beta > p.max_beta:
            raise ValueError("profile_1d_profilelib.min_beta > max_beta")
        if p.min_A > p.max_A:
            raise ValueError("profile_1d_profilelib.min_A > max_A")
        logger.info(
            "    Measuring peak shapes on reflections with I/sigma >= "
            f"{p.library.min_i_sigma} whose fit correlates >= "
            f"{p.library.min_correlation}, then holding them fixed elsewhere"
        )
        profile_1d_profilelib_params = TOFProfile1DProfilelibParams(
            {
                "A_min": p.min_A,
                "A_max": p.max_A,
                "alpha": p.init_alpha,
                "alpha_min": p.min_alpha,
                "alpha_max": p.max_alpha,
                "beta": p.init_beta,
                "beta_min": p.min_beta,
                "beta_max": p.max_beta,
                "n_restarts": p.n_restarts,
                "optimize_profile": True,
                "show_profile_failures": show_profile_failures,
                "peak_height_smoothing": p.peak_height_smoothing,
                "trust_min_corr": p.library.min_correlation,
                "trust_peak_tolerance": p.peak_tolerance,
                "trust_peak_height_fraction": p.peak_height_fraction,
                "library_min_i_sigma": p.library.min_i_sigma,
                "library_min_corr": p.library.min_correlation,
            }
        )
    elif params.method == "profile_1d_ic":
        p = params.profile_1d_ic
        # dict needed due to boost limit of 15 template arguments
        profile_1d_ic_params = TOFProfile1DICParams(
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
                "HatWidth": p.hat_width,
                "KConv": p.kconv,
                "n_restarts": p.n_restarts,
                "optimize_profile": True,
                "show_profile_failures": show_profile_failures,
            }
        )
    elif params.method == "profile_3d_gutmann":
        alpha = params.profile_3d_gutmann.init_alpha
        beta = params.profile_3d_gutmann.init_beta
        min_alpha = params.profile_3d_gutmann.min_alpha
        max_alpha = params.profile_3d_gutmann.max_alpha
        min_beta = params.profile_3d_gutmann.min_beta
        max_beta = params.profile_3d_gutmann.max_beta
        n_restarts = params.profile_3d_gutmann.n_restarts
        use_central_diff = (
            params.profile_3d_gutmann.gradient_method == "central_difference"
        )
        profile_3d_gutmann_params = TOFProfile3DGutmannParams(
            alpha,
            min_alpha,
            max_alpha,
            beta,
            min_beta,
            max_beta,
            n_restarts,
            True,
            use_central_diff,
            show_profile_failures,
        )
    elif params.method == "profile_3d_ic":
        p = params.profile_3d_ic
        # dict needed due to boost limit of 15 template arguments
        profile_3d_ic_params = TOFProfile3DICParams(
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
                "show_profile_failures": show_profile_failures,
            }
        )
    elif params.method == "profile_3d_ibix":
        p = params.profile_3d_ibix
        # dict needed due to boost limit of 15 template arguments
        profile_3d_ibix_params = TOFProfile3DIBIXParams(
            {
                "alpha": p.init_alpha,
                "alpha_min": p.min_alpha,
                "alpha_max": p.max_alpha,
                "beta": p.init_beta,
                "beta_min": p.min_beta,
                "beta_max": p.max_beta,
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
                "show_profile_failures": show_profile_failures,
            }
        )

    if apply_lorentz:
        logger.info("    Adding Lorentz correction")

    incident_params, absorption_params = get_spectrum_correction_params(**kwargs)
    if incident_params is not None:
        logger.info("    Adding incident spectrum correction")
    if absorption_params is not None:
        logger.info("    Adding absorption correction")

    profile_fitter = integrate_reflection_table(
        expt_reflections,
        expt,
        expt_data,
        incident_params,
        absorption_params,
        apply_lorentz,
        params.mp.nproc,
        profile_1d_ibix_params,
        profile_1d_ic_params,
        profile_3d_gutmann_params,
        profile_3d_ic_params,
        profile_3d_ibix_params,
        profile_1d_profilelib_params,
        params.corrections.lorentz_wavelength_power,
    )

    return expt_reflections, profile_fitter


def compute_tof_centroids(
    expt: Experiment, expt_reflections: flex.reflection_table
) -> flex.reflection_table:
    """
    Calculates the observed centroids from the background subtracted foreground
    of each shoebox. Time of flight component of centroid is set to
    the peak position along the time of flight.
    """

    shoeboxes = expt_reflections["shoebox"]
    centroids = shoeboxes.centroid_foreground_minus_background()
    x, y, _ = centroids.px_position().parts()
    _, _, peak_tof = shoeboxes.peak_coordinates().parts()
    expt_reflections["xyzobs.px.value"] = flex.vec3_double(x, y, peak_tof)
    expt_reflections["xyzobs.px.variance"] = centroids.px_std_err_eq()

    # centroid_px_to_mm selects by experiment index, so map this experiment to 0
    expt_ids = deepcopy(expt_reflections["imageset_id"])
    expt_reflections["imageset_id"] = flex.int(len(expt_reflections), 0)
    expt_reflections.centroid_px_to_mm(ExperimentList([expt]))
    expt_reflections["imageset_id"] = expt_ids

    return expt_reflections


def remove_overlapping_reflections(
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    overlaps = reflections.find_overlaps()
    overlap_sel = flex.bool(len(reflections), False)
    for item in overlaps.edges():
        i0 = overlaps.source(item)
        i1 = overlaps.target(item)
        overlap_sel[i0] = True
        overlap_sel[i1] = True
    logger.info("Rejecting %i overlapping bounding boxes", overlap_sel.count(True))
    reflections = reflections.select(~overlap_sel)
    return reflections


def assign_overlap_ownership(reflections: flex.reflection_table) -> flex.bool:
    """
    Gives each voxel shared between two bounding boxes to the reflection whose
    predicted centroid is nearer, flagging it Overlapped in the other. To be
    called before the foreground masks are calculated, so that a data-driven
    mask cannot be captured by a stronger neighbour.

    Returns which reflections have a bounding box that overlaps another.
    """
    return tof_assign_overlap_ownership(reflections)


def mask_neighbour_foreground(
    reflections: flex.reflection_table, has_overlap: flex.bool
) -> Tuple[flex.size_t, flex.size_t]:
    """
    Flags each voxel in one reflection's foreground as Overlapped in every other
    shoebox that contains it, then sets the overlapped_fg and overlapped_bg
    flags from the result. To be called after the foreground masks are
    calculated.

    A reflection flagged overlapped_fg has a neighbour's voxels in its own
    foreground. It is not integrated: summation and profile fitting both
    refuse a foreground containing an Overlapped voxel.

    Returns the coupled pairs, as two arrays of reflection indices: those
    whose intensities cannot be measured independently of each other.
    """
    coupled = tof_mask_neighbour_foreground(reflections)

    sboxs = reflections["shoebox"]
    overlapped_fg = (
        sboxs.count_mask_values(MaskCode.Foreground | MaskCode.Overlapped) > 0
    )
    overlapped_bg = (
        sboxs.count_mask_values(MaskCode.Background | MaskCode.Overlapped) > 0
    )
    reflections.set_flags(overlapped_fg, reflections.flags.overlapped_fg)
    reflections.set_flags(overlapped_bg, reflections.flags.overlapped_bg)

    n_overlap = has_overlap.count(True)
    n_fg = overlapped_fg.count(True)
    logger.info(
        f"    {n_overlap} of {len(reflections)} bounding boxes overlap another:"
        f" {n_overlap - n_fg} kept with the shared voxels masked,"
        f" {n_fg} with an overlapped foreground"
    )
    return coupled


def load_reference_intensities(
    path: str,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    """
    Reads merged intensities and gives each reflection that of its unique
    reflection, as intensity.reference.value and intensity.reference.variance.
    A variance of -1 marks a reflection the reference does not hold.

    The reference is on its own scale; compute_overlap_priors brings it to the
    scale of each experiment.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == ".refl":
        # TODO: accept a scaled DIALS reflection table. Merge the symmetry
        # equivalents of its scaled intensities (intensity.scale.value over
        # inverse_scale_factor), leaving out any flagged overlapped_fg, into
        # the mean and variance per unique reflection that a merged MTZ
        # supplies, and continue as below.
        raise NotImplementedError(
            "overlaps.reference does not yet accept a reflection table;"
            " give a merged .mtz"
        )
    if extension != ".mtz":
        raise ValueError(f"overlaps.reference must be a merged .mtz file, not '{path}'")

    from cctbx import miller
    from iotbx import mtz

    arrays = mtz.object(path).as_miller_arrays()
    reference = None
    for array in arrays:
        if array.is_xray_intensity_array() and array.sigmas() is not None:
            reference = array
            break
    if reference is None:
        for array in arrays:
            if array.is_xray_amplitude_array() and array.sigmas() is not None:
                label = array.info().label_string()
                reference = array.f_as_f_sq().set_info(array.info())
                logger.info(f"Squaring the amplitudes {label} of {path}")
                break
    if reference is None:
        raise ValueError(f"No intensities or amplitudes with sigmas found in {path}")
    label = reference.info().label_string()

    if reference.anomalous_flag():
        reference = reference.average_bijvoet_mates()
    reference = reference.map_to_asu()
    if not reference.is_unique_set_under_symmetry():
        reference = reference.merge_equivalents().array()

    point_group = experiments[0].crystal.get_space_group().build_derived_point_group()
    if reference.space_group().build_derived_point_group() != point_group:
        logger.warning(
            f"The space group of {path}"
            f" ({reference.space_group_info()}) and of the experiments"
            f" ({experiments[0].crystal.get_space_group().info()})"
            " differ in point group"
        )

    indices = miller.set(
        crystal_symmetry=reference.crystal_symmetry(),
        indices=reflections["miller_index"],
        anomalous_flag=False,
    ).map_to_asu()
    pairs = miller.match_multi_indices(
        miller_indices_unique=reference.indices(), miller_indices=indices.indices()
    ).pairs()
    i_reference = pairs.column(0)
    i_reflection = pairs.column(1)

    value = flex.double(len(reflections), 0.0)
    variance = flex.double(len(reflections), -1.0)
    value.set_selected(i_reflection, reference.data().select(i_reference).as_double())
    variance.set_selected(
        i_reflection, flex.pow2(reference.sigmas().select(i_reference).as_double())
    )
    reflections["intensity.reference.value"] = value
    reflections["intensity.reference.variance"] = variance
    logger.info(
        f"Reference intensities {label} from {path}:"
        f" {(variance > 0).count(True)} of {len(reflections)} reflections matched"
    )
    return reflections


def find_overlap_clusters(
    n_reflections: int, coupled_pairs: Tuple[flex.size_t, flex.size_t]
) -> flex.int:
    """
    Groups reflections into clusters whose intensities have to be found
    together: the connected components of the coupled pairs. Returns a cluster
    number per reflection, -1 for one that is in no cluster.
    """
    cluster_id = np.full(n_reflections, -1, dtype=np.int32)
    first = flumpy.to_numpy(coupled_pairs[0]).astype(np.int64)
    second = flumpy.to_numpy(coupled_pairs[1]).astype(np.int64)
    if len(first) == 0:
        return flumpy.from_numpy(cluster_id)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix(
        (np.ones(len(first)), (first, second)), shape=(n_reflections, n_reflections)
    )
    _, labels = connected_components(graph, directed=False)
    member = np.zeros(n_reflections, dtype=bool)
    member[first] = True
    member[second] = True
    _, compact = np.unique(labels[member], return_inverse=True)
    cluster_id[member] = compact
    return flumpy.from_numpy(cluster_id)


def compute_overlap_priors(
    reflections: flex.reflection_table, params: libtbx.phil.scope_extract
) -> bool:
    """
    Turns the reference intensities into a prior on the intensity of each
    reflection of one experiment, as overlap.prior.value and
    overlap.prior.variance. A variance of -1 means no prior.

    The reference is merged and scaled; an integrated intensity is neither, and
    depends on wavelength through the incident spectrum and whatever part of
    the Lorentz factor was applied. The two are related here by a scale
    k(wavelength), the ratio of summed intensities in bins of wavelength, taken
    from the reflections of this experiment that were integrated without
    overlap. Their relative scatter f about the scaled reference, per bin, says
    how well one observation is predicted by it, so the prior for a reflection
    with reference intensity I and variance v is

        mean = k I,   variance = k^2 v + (f k I)^2

    widened by overlaps.prior_sigma_scale. f includes the measurement error of
    the reflections it is taken from, so the prior errs on the wide side.

    Returns False, leaving every reflection without a prior, if there is no
    reference or it does not correlate with this experiment's intensities.
    """
    n = len(reflections)
    reflections["overlap.prior.value"] = flex.double(n, 0.0)
    reflections["overlap.prior.variance"] = flex.double(n, -1.0)
    if "intensity.reference.value" not in reflections:
        return False

    i_ref = flumpy.to_numpy(reflections["intensity.reference.value"])
    var_ref = flumpy.to_numpy(reflections["intensity.reference.variance"])
    i_raw = flumpy.to_numpy(reflections["intensity.prf.value"])
    var_raw = flumpy.to_numpy(reflections["intensity.prf.variance"])
    wavelength = flumpy.to_numpy(reflections["wavelength_cal"])
    flags = reflections.flags
    integrated = flumpy.to_numpy(reflections.get_flags(flags.integrated_prf))
    overlapped = flumpy.to_numpy(reflections.get_flags(flags.overlapped_fg))
    has_ref = var_ref > 0
    clean = integrated & ~overlapped & has_ref & np.isfinite(i_raw) & (var_raw > 0)

    n_clean = int(clean.sum())
    if n_clean < 10:
        logger.info(
            f"    Only {n_clean} unoverlapped reflections match the reference:"
            " no intensity prior"
        )
        return False

    # Bins of equal population in wavelength
    n_bins = max(1, min(params.overlaps.n_wavelength_bins, n_clean // 20))
    edges = np.quantile(wavelength[clean], np.linspace(0.0, 1.0, n_bins + 1))
    which = np.clip(
        np.searchsorted(edges, wavelength[clean], side="right") - 1, 0, n_bins - 1
    )
    centre = np.full(n_bins, np.nan)
    scale = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            continue
        numerator = i_raw[clean][sel].sum()
        denominator = i_ref[clean][sel].sum()
        if numerator > 0 and denominator > 0:
            centre[b] = np.median(wavelength[clean][sel])
            scale[b] = numerator / denominator
    good = np.isfinite(scale)
    if not good.any():
        logger.info("    No scale to the reference could be found: no intensity prior")
        return False
    k = np.interp(wavelength, centre[good], scale[good])

    predicted = k[clean] * i_ref[clean]
    cc = float(np.corrcoef(i_raw[clean], predicted)[0, 1])
    if not cc >= params.overlaps.min_reference_cc:
        logger.warning(
            f"    The reference correlates {cc:.3f} with this experiment's"
            f" unoverlapped intensities, below overlaps.min_reference_cc="
            f"{params.overlaps.min_reference_cc}: no intensity prior."
            " Is it indexed consistently with the experiments?"
        )
        return False

    # Relative scatter of one strong observation about the scaled reference
    strong = (
        (i_raw[clean] > 5.0 * np.sqrt(var_raw[clean]))
        & (i_ref[clean] > 5.0 * np.sqrt(var_ref[clean]))
        & (predicted > 0)
    )

    def robust_scatter(ratio):
        return 1.4826 * np.median(np.abs(ratio - np.median(ratio)))

    ratio = i_raw[clean][strong] / predicted[strong]
    if len(ratio) >= 5:
        overall = robust_scatter(ratio)
    else:
        ratio_all = i_raw[clean][predicted > 0] / predicted[predicted > 0]
        overall = robust_scatter(ratio_all) if len(ratio_all) else 1.0
    scatter = np.full(n_bins, overall)
    for b in range(n_bins):
        sel = which[strong] == b
        if sel.sum() >= 5:
            scatter[b] = robust_scatter(ratio[sel])
    f = np.interp(wavelength, centre[good], scatter[good])

    mean = k * i_ref
    variance = params.overlaps.prior_sigma_scale**2 * (
        k**2 * np.where(has_ref, var_ref, 0.0) + (f * mean) ** 2
    )
    usable = has_ref & np.isfinite(mean) & np.isfinite(variance) & (variance > 0)
    reflections["overlap.prior.value"] = flumpy.from_numpy(np.where(usable, mean, 0.0))
    reflections["overlap.prior.variance"] = flumpy.from_numpy(
        np.where(usable, variance, -1.0)
    )
    logger.info(
        f"    Intensity prior from {n_clean} unoverlapped reflections in"
        f" {int(good.sum())} wavelength bins: CC to the scaled reference"
        f" {cc:.3f}, scatter of one observation about it {np.median(scatter[good]):.0%}"
    )
    return True


def calibrate_joint_fit(
    expt: Experiment,
    expt_reflections: flex.reflection_table,
    expt_data: ImageSequence,
    params: libtbx.phil.scope_extract,
    profile_fitter: object,
    incident_params: object,
    absorption_params: object,
    max_sample: int = 2000,
) -> float:
    """
    The scale of an intensity fitted in 3D relative to one fitted to the ToF
    projection, as their median ratio over strong reflections that are in no
    cluster, each put through the joint fit on its own.

    The two fits extrapolate the tails of a peak differently, so they do not
    return the same number for the same reflection. Without this the
    deconvolved reflections would sit on a different scale from the rest of
    the table. The spread of the ratio is logged: it is a direct measure of how
    well the library's 3D shapes describe reflections whose intensity is known.

    Returns 1 if too few reflections qualify.
    """
    flags = expt_reflections.flags
    i_prf = expt_reflections["intensity.prf.value"]
    var_prf = expt_reflections["intensity.prf.variance"]
    min_i_sigma = params.profile_1d_profilelib.library.min_i_sigma
    sel = (
        expt_reflections.get_flags(flags.integrated_prf)
        & ~expt_reflections.get_flags(flags.overlapped_fg)
        & (expt_reflections["overlap.cluster_id"] < 0)
        & (var_prf > 0)
    )
    sel = sel & (i_prf > min_i_sigma * flex.sqrt(flex.abs(var_prf)))
    indices = sel.iselection()
    if len(indices) > max_sample:
        step = len(indices) / max_sample
        indices = indices.select(
            flex.size_t([int(i * step) for i in range(max_sample)])
        )
    if len(indices) < 20:
        logger.info(
            "    Too few strong unoverlapped reflections to calibrate the"
            " joint fit against: its intensities are used as they are"
        )
        return 1.0

    sample = expt_reflections.select(indices)
    n = len(sample)
    sample["overlap.cluster_id"] = flex.int(range(n))
    sample["overlap.prior.value"] = flex.double(n, 0.0)
    sample["overlap.prior.variance"] = flex.double(n, -1.0)
    tof_deconvolve_overlaps(
        sample,
        expt,
        expt_data,
        incident_params,
        absorption_params,
        params.corrections.lorentz,
        params.corrections.lorentz_wavelength_power,
        params.mp.nproc,
        profile_fitter,
    )
    fitted = flumpy.to_numpy(sample["overlap.ml.variance"]) > 0
    if fitted.sum() < 20:
        logger.info(
            "    Too few strong unoverlapped reflections could be fitted in 3D"
            " to calibrate the joint fit against: its intensities are used as"
            " they are"
        )
        return 1.0
    joint = flumpy.to_numpy(sample["overlap.ml.value"])[fitted]
    own = flumpy.to_numpy(sample["intensity.prf.value"])[fitted]
    ratio = joint / own
    scale = float(np.median(ratio))
    if not (scale > 0 and np.isfinite(scale)):
        return 1.0
    q25, q75 = np.percentile(ratio, [25, 75])
    logger.info(
        f"    3D fit of {int(fitted.sum())} strong unoverlapped reflections"
        f" against their own profile-fitted intensity: CC"
        f" {np.corrcoef(joint, own)[0, 1]:.3f}, ratio {scale:.2f}"
        f" (quartiles {q25:.2f}, {q75:.2f}). Joint fits are divided by {scale:.2f}"
    )
    return scale


def deconvolve_overlaps(
    expt: Experiment,
    expt_reflections: flex.reflection_table,
    expt_data: ImageSequence,
    params: libtbx.phil.scope_extract,
    profile_fitter: object,
    coupled_pairs: Tuple[flex.size_t, flex.size_t],
    **kwargs: Dict,
) -> flex.reflection_table:
    """
    Fits each cluster of overlapped reflections jointly, with shapes from the
    profile library and the intensity prior from compute_overlap_priors.

    A reflection flagged overlapped_fg takes the posterior as
    intensity.prf.value and intensity.prf.variance and gains integrated_prf.
    Every member of a fitted cluster gets overlap.fit.value / .variance (the
    posterior) and overlap.ml.value / .variance (the data alone), which with
    overlap.prior.value / .variance show how far the prior moved it. Variances
    of -1 mark values that are absent.
    """
    n = len(expt_reflections)
    cluster_id = find_overlap_clusters(n, coupled_pairs)
    expt_reflections["overlap.cluster_id"] = cluster_id
    compute_overlap_priors(expt_reflections, params)
    for name in ("overlap.fit", "overlap.ml"):
        expt_reflections[name + ".value"] = flex.double(n, 0.0)
        expt_reflections[name + ".variance"] = flex.double(n, -1.0)
    expt_reflections["overlap.fit.chi2"] = flex.double(n, -1.0)
    expt_reflections["overlap.fit.fraction"] = flex.double(n, 0.0)

    in_cluster = cluster_id >= 0
    if in_cluster.count(True) == 0:
        logger.info("    No overlapped reflections to deconvolve")
        return expt_reflections
    if profile_fitter is None or profile_fitter.library_size_xy() == 0:
        logger.warning(
            "    The profile library holds no peak shapes on the detector face:"
            " overlapped reflections are left unintegrated"
        )
        return expt_reflections

    incident_params, absorption_params = get_spectrum_correction_params(**kwargs)
    intensity_scale = calibrate_joint_fit(
        expt,
        expt_reflections,
        expt_data,
        params,
        profile_fitter,
        incident_params,
        absorption_params,
    )
    members = expt_reflections.select(in_cluster)
    n_fitted = tof_deconvolve_overlaps(
        members,
        expt,
        expt_data,
        incident_params,
        absorption_params,
        params.corrections.lorentz,
        params.corrections.lorentz_wavelength_power,
        params.mp.nproc,
        profile_fitter,
        intensity_scale,
    )
    expt_reflections.set_selected(in_cluster, members)
    log_deconvolution_summary(expt_reflections, n_fitted)
    return expt_reflections


def fit_profiles_jointly(
    expt: Experiment,
    expt_reflections: flex.reflection_table,
    expt_data: ImageSequence,
    params: libtbx.phil.scope_extract,
    profile_fitter: object,
    coupled_pairs: Optional[Tuple[flex.size_t, flex.size_t]],
    **kwargs: Dict,
) -> flex.reflection_table:
    """
    overlaps.estimator=joint: every reflection is fitted with the 3D model of
    tof_deconvolve_overlaps, so that every profile-fitted intensity comes from
    the same estimator and no scale between two fits is needed.

    With coupled_pairs (overlaps.mode=deconvolve) the overlapped reflections
    are fitted in their clusters, with the intensity prior if a reference was
    given. Every other reflection that can be integrated is a cluster of its
    own and has no prior: a reflection measured without overlap is not pulled
    towards its equivalents.

    The 1D profile fit is kept as intensity.prf_1d.value / .variance (variance
    -1 where it failed). A reflection the 3D model cannot fit loses its
    profile-fitted intensity rather than keep the 1D one, which would mix the
    two estimators again.
    """
    n = len(expt_reflections)
    flags = expt_reflections.flags
    prf_1d = expt_reflections.get_flags(flags.integrated_prf)
    expt_reflections["intensity.prf_1d.value"] = expt_reflections[
        "intensity.prf.value"
    ].deep_copy()
    variance_1d = expt_reflections["intensity.prf.variance"].deep_copy()
    variance_1d.set_selected(~prf_1d, -1.0)
    expt_reflections["intensity.prf_1d.variance"] = variance_1d

    if profile_fitter is None or profile_fitter.library_size_xy() == 0:
        logger.warning(
            "    The profile library holds no peak shapes on the detector face:"
            " intensities are left from the 1D profile fit"
        )
        return expt_reflections

    if coupled_pairs is not None:
        cluster_id = find_overlap_clusters(n, coupled_pairs)
        expt_reflections["overlap.cluster_id"] = cluster_id
        compute_overlap_priors(expt_reflections, params)
    else:
        cluster_id = flex.int(n, -1)
        expt_reflections["overlap.prior.value"] = flex.double(n, 0.0)
        expt_reflections["overlap.prior.variance"] = flex.double(n, -1.0)

    # Everything else that can be integrated is a cluster of one, without a
    # prior
    candidates = (
        (cluster_id < 0)
        & ~expt_reflections.get_flags(flags.overlapped_fg)
        & ~expt_reflections.get_flags(flags.dont_integrate)
        & ~expt_reflections.get_flags(flags.failed_during_background_modelling)
    )
    singles = candidates.iselection()
    first = flex.max(cluster_id) + 1 if (cluster_id >= 0).count(True) else 0
    cluster_id.set_selected(singles, flex.int(range(first, first + len(singles))))
    prior_variance = expt_reflections["overlap.prior.variance"]
    prior_variance.set_selected(singles, -1.0)
    expt_reflections["overlap.prior.variance"] = prior_variance
    expt_reflections["overlap.cluster_id"] = cluster_id
    for name in ("overlap.fit", "overlap.ml"):
        expt_reflections[name + ".value"] = flex.double(n, 0.0)
        expt_reflections[name + ".variance"] = flex.double(n, -1.0)
    expt_reflections["overlap.fit.chi2"] = flex.double(n, -1.0)
    expt_reflections["overlap.fit.fraction"] = flex.double(n, 0.0)

    # A reflection outside every cluster keeps no 1D intensity either
    outside = (cluster_id < 0) & prf_1d
    expt_reflections.unset_flags(outside, flags.integrated_prf)
    expt_reflections.set_flags(outside, flags.failed_during_profile_fitting)

    incident_params, absorption_params = get_spectrum_correction_params(**kwargs)
    in_cluster = cluster_id >= 0
    members = expt_reflections.select(in_cluster)
    n_fitted = tof_deconvolve_overlaps(
        members,
        expt,
        expt_data,
        incident_params,
        absorption_params,
        params.corrections.lorentz,
        params.corrections.lorentz_wavelength_power,
        params.mp.nproc,
        profile_fitter,
        1.0,
        True,
    )
    expt_reflections.set_selected(in_cluster, members)

    prf = expt_reflections.get_flags(flags.integrated_prf)
    logger.info(
        f"    3D fit of {in_cluster.count(True)} reflections"
        f" ({len(singles)} on their own): {prf.count(True)} given an intensity,"
        f" {(prf_1d & ~prf).count(True)} that the 1D fit had integrated were not"
    )
    chi2 = flumpy.to_numpy(expt_reflections["overlap.fit.chi2"])
    chi2 = chi2[chi2 > 0]
    if len(chi2):
        logger.info(
            f"    Reduced chi-squared over the peaks: median {np.median(chi2):.2f},"
            f" above 1 (variance inflated by it) for {np.mean(chi2 > 1):.0%},"
            f" above 2 for {np.mean(chi2 > 2):.0%}"
        )
    log_deconvolution_summary(expt_reflections, n_fitted)
    return expt_reflections


def log_deconvolution_summary(
    reflections: flex.reflection_table, n_fitted: int
) -> None:
    """
    Reports what the joint fits did, and the two checks on them that need
    nothing but this experiment: how far the prior moved the intensities, and
    how the joint fit compares with the intensities of the cluster members
    that could also be integrated on their own.
    """
    cluster_id = flumpy.to_numpy(reflections["overlap.cluster_id"])
    fit = flumpy.to_numpy(reflections["overlap.fit.value"])
    fit_var = flumpy.to_numpy(reflections["overlap.fit.variance"])
    ml = flumpy.to_numpy(reflections["overlap.ml.value"])
    ml_var = flumpy.to_numpy(reflections["overlap.ml.variance"])
    prior = flumpy.to_numpy(reflections["overlap.prior.value"])
    prior_var = flumpy.to_numpy(reflections["overlap.prior.variance"])
    flags = reflections.flags
    overlapped = flumpy.to_numpy(reflections.get_flags(flags.overlapped_fg))

    n_clusters = len(np.unique(cluster_id[cluster_id >= 0]))
    fitted = fit_var > 0
    recovered = fitted & overlapped
    logger.info(
        f"    Fitted {n_fitted} of {n_clusters} clusters:"
        f" {int(recovered.sum())} of {int(overlapped.sum())} overlapped"
        " reflections given an intensity"
    )

    sel = recovered & (prior_var > 0)
    if sel.any():
        pull = (fit[sel] - prior[sel]) / np.sqrt(prior_var[sel])
        data_weight = 1.0 - fit_var[sel] / prior_var[sel]
        message = (
            f"    With a prior ({int(sel.sum())}): median weight of the data in"
            f" the posterior {np.median(data_weight):.2f}, posterior minus prior"
            f" in prior sigmas: median {np.median(pull):+.2f},"
            f" rms {np.sqrt(np.mean(pull**2)):.2f}"
        )
        both = sel & (ml_var > 0)
        if both.any():
            shift = (fit[both] - ml[both]) / np.sqrt(ml_var[both])
            message += (
                ". Moved by the prior, in sigmas of the data alone: median"
                f" {np.median(np.abs(shift)):.2f}"
            )
        logger.info(message)

    if "intensity.prf_1d.value" in reflections:
        i_prf = flumpy.to_numpy(reflections["intensity.prf_1d.value"])
        var_prf = flumpy.to_numpy(reflections["intensity.prf_1d.variance"])
        integrated = var_prf > 0
    else:
        integrated = flumpy.to_numpy(reflections.get_flags(flags.integrated_prf))
        i_prf = flumpy.to_numpy(reflections["intensity.prf.value"])
        var_prf = flumpy.to_numpy(reflections["intensity.prf.variance"])
    sel = fitted & ~overlapped & integrated & (var_prf > 0) & (ml_var > 0)
    if sel.sum() >= 5:
        z = (ml[sel] - i_prf[sel]) / np.sqrt(ml_var[sel] + var_prf[sel])
        logger.info(
            f"    Check on {int(sel.sum())} reflections also fitted in 1D:"
            " 3D fit without prior against the 1D profile fit,"
            f" CC {np.corrcoef(ml[sel], i_prf[sel])[0, 1]:.3f}, median ratio"
            f" {np.median(ml[sel] / i_prf[sel]):.2f}, rms difference"
            f" {np.sqrt(np.mean(z**2)):.2f} sigma"
        )


def remove_out_of_range_reflections(
    reflections: flex.reflection_table, params: libtbx.phil.scope_extract
) -> flex.reflection_table:

    if params.wavelength_range is not None:
        min_wavelength = float(params.wavelength_range[0])
        max_wavelength = float(params.wavelength_range[1])
        wavelength_sel = (reflections["wavelength_cal"] >= min_wavelength) & (
            reflections["wavelength_cal"] <= max_wavelength
        )
        logger.info(
            f"Removing {wavelength_sel.count(False)} reflections outside of range"
            f" ({min_wavelength}, {max_wavelength}) (A)"
        )
        reflections = reflections.select(wavelength_sel)

    if params.tof_range is not None:
        min_tof = float(params.tof_range[0])
        max_tof = float(params.tof_range[1])
        tof_sel = (reflections["tof_cal"] >= min_tof) & (
            reflections["tof_cal"] <= max_tof
        )
        logger.info(
            f"Removing {tof_sel.count(False)} reflections outside of range"
            f" ({min_tof}, {max_tof}) (usec)"
        )
        reflections = reflections.select(tof_sel)

    assert len(reflections) > 0, (
        "No reflections left after filtering wavelength/tof range"
    )
    return reflections


def compute_partiality(bbox: Tuple, image_size: Tuple) -> float:
    """
    Approximate partiality by looking at overlap of bounding box with panel
    """

    intersect_x0 = max(image_size[0], bbox[0])
    intersect_y0 = max(image_size[2], bbox[2])
    intersect_x1 = min(image_size[1], bbox[1])
    intersect_y1 = min(image_size[3], bbox[3])

    if intersect_x0 >= intersect_x1 or intersect_y0 >= intersect_y1:
        return 0

    intersection_area = (intersect_x1 - intersect_x0) * (intersect_y1 - intersect_y0)
    square_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])

    return intersection_area / square_area


def update_bounding_box(
    bbox: Tuple,
    centroid: Tuple,
    new_centroid: Tuple,
    padding: Tuple,  # x,y,z
    image_size: Tuple,
) -> Tuple[Tuple, float]:
    """
    Calculates a new bounding box originally at centroid at position
    new_centroid with padding.
    Returns the bounding box and its partiality
    """

    diff_centroid = (
        new_centroid[0] - centroid[0],
        new_centroid[1] - centroid[1],
        new_centroid[2] - centroid[2],
    )

    updated_bbox = list(deepcopy(bbox))

    updated_bbox[0] += diff_centroid[0] - padding[0]
    updated_bbox[1] += diff_centroid[0] + padding[0]
    updated_bbox[2] += diff_centroid[1] - padding[1]
    updated_bbox[3] += diff_centroid[1] + padding[1]
    updated_bbox[4] += diff_centroid[2] - padding[2]
    updated_bbox[5] += diff_centroid[2] + padding[2]

    partiality = compute_partiality(updated_bbox, image_size)

    updated_bbox[0] = max(floor(updated_bbox[0]), image_size[0])
    updated_bbox[1] = min(ceil(updated_bbox[1]), image_size[1])
    updated_bbox[2] = max(floor(updated_bbox[2]), image_size[2])
    updated_bbox[3] = min(ceil(updated_bbox[3]), image_size[3])
    updated_bbox[4] = max(floor(updated_bbox[4]), image_size[4])
    updated_bbox[5] = min(ceil(updated_bbox[5]), image_size[5])

    return tuple(updated_bbox), partiality


def get_predicted_observed_reflections(
    params: libtbx.phil.scope_extract,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    """
    Returns predicted reflections matched to reflections
    """

    # Calculate dmin
    min_s0_idx = min(
        range(len(reflections["wavelength"])), key=reflections["wavelength"].__getitem__
    )
    min_s0 = reflections["s0"][min_s0_idx]
    dmin = None
    for experiment in experiments:
        expt_dmin = experiment.detector.get_max_resolution(min_s0)
        if dmin is None or expt_dmin < dmin:
            dmin = expt_dmin

    logger.info(f"Calculated dmin from observed reflections: {round(dmin, 3)}")

    predicted_reflections = None
    miller_indices = reflections["miller_index"]

    # Get predicted reflections for observed miller indices
    for idx, experiment in enumerate(experiments):
        predictor = TOFReflectionPredictor(experiment, dmin)
        if predicted_reflections is None:
            predicted_reflections = predictor.indices_for_ub(miller_indices)
            predicted_reflections["id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
            predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
        else:
            r = predictor.indices_for_ub(miller_indices)
            r["id"] = cctbx.array_family.flex.int(len(r), idx)
            r["imageset_id"] = cctbx.array_family.flex.int(len(r), idx)
            predicted_reflections.extend(r)

    # Update params
    predicted_reflections["s0"] = predicted_reflections["s0_cal"]
    # Add predicted time of flight (usec) to xyzcal.mm
    x_cal, y_cal, _ = predicted_reflections["xyzcal.mm"].parts()
    predicted_reflections["xyzcal.mm"] = flex.vec3_double(
        x_cal, y_cal, predicted_reflections["tof_cal"]
    )
    predicted_reflections.calculate_entering_flags(experiments)

    for i in range(len(experiments)):
        predicted_reflections.experiment_identifiers()[i] = experiments[i].identifier

    # Updates flags to set which reflections to use in generating reference profiles
    matched, reflections, _ = predicted_reflections.match_with_reference(reflections)

    predicted_reflections = predicted_reflections.select(matched)
    if "idx" in reflections:
        predicted_reflections["idx"] = reflections["idx"]

    predicted_reflections["xyzobs.px.value"] = reflections["xyzobs.px.value"]

    # Calculate predicted bounding boxes and partiality
    tof_padding = params.bbox_tof_padding
    xy_padding = params.bbox_xy_padding
    image_size = experiments[0].detector[0].get_image_size()
    tof_size = len(experiments[0].scan.get_property("time_of_flight"))
    bboxes = flex.int6(len(predicted_reflections))
    partiality = flex.double(len(predicted_reflections))
    for i in range(len(predicted_reflections)):
        bboxes[i], partiality[i] = update_bounding_box(
            reflections["bbox"][i],
            reflections["xyzobs.px.value"][i],
            predicted_reflections["xyzcal.px"][i],
            (int(xy_padding), int(xy_padding), int(tof_padding)),
            (0, image_size[0], 0, image_size[1], 0, tof_size),
        )
    predicted_reflections["bbox"] = bboxes
    predicted_reflections["partiality"] = partiality

    predicted_reflections.compute_d(experiments)
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )

    predicted_reflections.map_centroids_to_reciprocal_space(
        experiments, calculated=True
    )
    return predicted_reflections


def get_predicted_calculated_reflections(
    params: libtbx.phil.scope_extract,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    """
    Returns predicted reflections out to params.calculated.dmin
    """

    dmin = params.calculated.dmin
    assert dmin is not None, (
        "Integrating calculated reflections but calculated.dmin has not been set"
    )

    # Get predicted reflections
    predicted_reflections = None
    for idx, experiment in enumerate(experiments):
        predictor = TOFReflectionPredictor(experiment, dmin)
        if predicted_reflections is None:
            predicted_reflections = predictor.for_ub(experiment.crystal.get_A())
            predicted_reflections["id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
            predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
        else:
            r = predictor.for_ub(experiment.crystal.get_A())
            r["id"] = cctbx.array_family.flex.int(len(r), idx)
            r["imageset_id"] = cctbx.array_family.flex.int(len(r), idx)
            predicted_reflections.extend(r)
    predicted_reflections["s0"] = predicted_reflections["s0_cal"]
    # Add predicted time of flight (usec) to xyzcal.mm
    x_cal, y_cal, _ = predicted_reflections["xyzcal.mm"].parts()
    predicted_reflections["xyzcal.mm"] = flex.vec3_double(
        x_cal, y_cal, predicted_reflections["tof_cal"]
    )
    predicted_reflections.calculate_entering_flags(experiments)

    for i in range(len(experiments)):
        predicted_reflections.experiment_identifiers()[i] = experiments[i].identifier

    logger.info(f"Predicted {len(predicted_reflections)} using dmin {dmin}")

    # Updates flags to set which reflections were observed
    predicted_reflections.match_with_reference(reflections)

    used_in_ref = reflections.get_flags(reflections.flags.used_in_refinement)
    model_reflections = reflections.select(used_in_ref)
    logger.info(
        f"Using {len(model_reflections)} observed reflections"
        " to calculate bounding boxes"
    )

    # Get the closest observed reflection for each calculated reflection
    # Use the observed bounding box as a basis for the calculated reflection

    tof_padding = params.bbox_tof_padding
    xy_padding = params.bbox_xy_padding
    image_size = experiments[0].detector[0].get_image_size()
    tof_size = len(experiments[0].scan.get_property("time_of_flight"))
    predicted_reflections["bbox"] = flex.int6(len(predicted_reflections))
    predicted_reflections["partiality"] = flex.double(len(predicted_reflections))

    # If no observed reflections on a given panel, use the average bbox size
    x0, x1, y0, y1, z0, z1 = reflections["bbox"].parts()
    num_reflections = len(reflections)
    avg_bbox_size = (
        (sum(x1 - x0) / num_reflections) * 0.5,
        (sum(y1 - y0) / num_reflections) * 0.5,
        (sum(z1 - z0) / num_reflections) * 0.5,
    )

    for panel_idx in range(len(experiments[0].detector)):
        for expt_idx in range(len(experiments)):
            p_sel = (predicted_reflections["id"] == expt_idx) & (
                predicted_reflections["panel"] == panel_idx
            )
            expt_p_reflections = predicted_reflections.select(p_sel)
            bboxes = flex.int6(len(expt_p_reflections))
            partiality = flex.double(len(expt_p_reflections))

            m_sel = (model_reflections["id"] == expt_idx) & (
                model_reflections["panel"] == panel_idx
            )
            expt_m_reflections = model_reflections.select(m_sel)

            for i in range(len(expt_p_reflections)):
                if len(expt_m_reflections) == 0:
                    c = expt_p_reflections["xyzcal.px"][i]
                    bbox = (
                        c[0] - avg_bbox_size[0],
                        c[0] + avg_bbox_size[0],
                        c[1] - avg_bbox_size[1],
                        c[1] + avg_bbox_size[1],
                        c[2] - avg_bbox_size[2],
                        c[2] + avg_bbox_size[2],
                    )

                    bboxes[i], partiality[i] = update_bounding_box(
                        bbox,
                        expt_p_reflections["xyzcal.px"][i],
                        expt_p_reflections["xyzcal.px"][i],
                        (int(xy_padding), int(xy_padding), int(tof_padding)),
                        (0, image_size[0], 0, image_size[1], 0, tof_size),
                    )

                else:
                    distances = (
                        expt_m_reflections["xyzobs.px.value"]
                        - expt_p_reflections["xyzcal.px"][i]
                    ).norms()
                    min_distance_idx = np.argmin(flumpy.to_numpy(distances))

                    bboxes[i], partiality[i] = update_bounding_box(
                        expt_m_reflections["bbox"][min_distance_idx],
                        expt_m_reflections["xyzobs.px.value"][min_distance_idx],
                        expt_p_reflections["xyzcal.px"][i],
                        (int(xy_padding), int(xy_padding), int(tof_padding)),
                        (0, image_size[0], 0, image_size[1], 0, tof_size),
                    )
            predicted_reflections["bbox"].set_selected(p_sel, bboxes)
            predicted_reflections["partiality"].set_selected(p_sel, partiality)

    _, _, _, _, z1, z2 = predicted_reflections["bbox"].parts()
    sel = z2 > z1
    predicted_reflections = predicted_reflections.select(sel)
    _, _, y1, y2, _, _ = predicted_reflections["bbox"].parts()
    sel = y2 > y1
    predicted_reflections = predicted_reflections.select(sel)
    x1, x2, _, _, _, _ = predicted_reflections["bbox"].parts()
    sel = x2 > x1
    predicted_reflections = predicted_reflections.select(sel)

    predicted_reflections.compute_d(experiments)
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )

    predicted_reflections.map_centroids_to_reciprocal_space(
        experiments, calculated=True
    )
    return predicted_reflections


def run():
    """
    Input setup
    """

    phil = phil_scope.fetch()

    usage = "usage: dials.tof_integrate.py refined.expt refined.refl"
    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        epilog=__doc__,
        read_experiments=True,
        read_reflections=True,
    )

    args = _translate_deprecated_args(sys.argv[1:])
    params, options = parser.parse_args(args=args, show_diff_phil=False)

    dials.util.log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    """
    Load experiment and reflections
    """

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )
    reflections = reflections[0]

    integrated_reflections = run_integrate(params, experiments, reflections)
    integrated_reflections.as_msgpack_file(params.output.reflections)
    experiments.as_file(params.output.experiments)


def applying_spherical_absorption_correction(params: libtbx.phil.scope_extract) -> bool:
    """
    Returns True if all params are present for absorption correction.
    Raises error if only some are present
    """

    all_params_present = True
    some_params_present = False
    for i in dir(params.corrections.absorption.target_spectrum):
        if i.startswith("__"):
            continue
        if getattr(params.corrections.absorption.target_spectrum, i) is not None:
            some_params_present = True
        else:
            all_params_present = False
    if some_params_present and not all_params_present:
        raise ValueError(
            "Trying to apply spherical absorption correction but some corrections are None."
        )
    return all_params_present


def applying_incident_and_empty_runs(params: libtbx.phil.scope_extract) -> bool:
    """
    Returns True if all params are present for incident correction.
    Raises error if only some are present
    """

    if params.corrections.incident_run is not None:
        assert params.corrections.empty_run is not None, (
            "Incident run given without empty run."
        )
        return True
    elif params.corrections.empty_run is not None:
        assert params.corrections.incident_run is not None, (
            "Empty run given without incident run."
        )
        return True
    return False


def run_integrate(
    params: libtbx.phil.scope_extract,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    if params.mp.nproc is libtbx.Auto:
        params.mp.nproc = multiprocessing.cpu_count()

    if (
        params.overlaps.mode == "deconvolve"
        and params.method != "profile_1d_profilelib"
    ):
        raise ValueError(
            "overlaps.mode=deconvolve takes its peak shapes from the profile"
            " library, and requires method=profile_1d_profilelib"
        )
    if (
        params.overlaps.estimator == "joint"
        and params.method != "profile_1d_profilelib"
    ):
        raise ValueError(
            "overlaps.estimator=joint takes its peak shapes from the profile"
            " library, and requires method=profile_1d_profilelib"
        )

    reflections, _ = process_reference(reflections)

    """
    Predict reflections using experiment crystal
    """

    if params.integration_type == "observed":
        predicted_reflections = get_predicted_observed_reflections(
            params=params, experiments=experiments, reflections=reflections
        )
    elif params.integration_type == "calculated":
        predicted_reflections = get_predicted_calculated_reflections(
            params=params, experiments=experiments, reflections=reflections
        )
    else:
        raise NotImplementedError(f"Unknown integration type {params.integration_type}")

    if params.overlaps.mode == "reject":
        predicted_reflections = remove_overlapping_reflections(predicted_reflections)
    predicted_reflections = remove_out_of_range_reflections(
        predicted_reflections, params
    )

    if params.overlaps.reference is not None:
        if params.overlaps.mode == "deconvolve":
            predicted_reflections = load_reference_intensities(
                params.overlaps.reference, experiments, predicted_reflections
            )
        else:
            logger.warning(
                "overlaps.reference is only used with overlaps.mode=deconvolve"
            )
    elif params.overlaps.mode == "deconvolve":
        logger.info(
            "No overlaps.reference given: overlapped reflections will be"
            " separated by shape alone"
        )

    corrections_data = get_corrections_data(experiments=experiments, params=params)

    experiment_cls = experiments[0].imageset.get_format_class()
    for idx, expt in enumerate(experiments):
        logger.info(f"Integrating experiment {idx}")

        expt_data = expt.imageset

        if "incident_proton_charge" in corrections_data:
            expt_proton_charge = experiment_cls.get_instance(
                expt.imageset.paths()[0], **expt.imageset.data().get_params()
            ).get_proton_charge()
            corrections_data["expt_proton_charge"] = expt_proton_charge

        sel_expt = predicted_reflections["id"] == idx
        expt_reflections = predicted_reflections.select(sel_expt)

        logger.info("    Extracting shoeboxes")
        tof_extract_shoeboxes_to_reflection_table(
            expt_reflections, expt, expt_data, False
        )

        if params.overlaps.mode == "reject":
            expt_reflections = calculate_shoebox_masks(expt, expt_reflections, params)
        else:
            has_overlap = assign_overlap_ownership(expt_reflections)
            expt_reflections = calculate_shoebox_masks(expt, expt_reflections, params)
            coupled_pairs = mask_neighbour_foreground(expt_reflections, has_overlap)
        expt_reflections.is_overloaded(experiments)
        expt_reflections.contains_invalid_pixels()

        # Filter reflections with a high fraction of masked foreground
        valid_foreground_threshold = 1.0  # DIALS default
        sboxs = expt_reflections["shoebox"]
        nvalfg = sboxs.count_mask_values(MaskCode.Valid | MaskCode.Foreground)
        nforeg = sboxs.count_mask_values(MaskCode.Foreground)
        fraction_valid = nvalfg.as_double() / nforeg.as_double()
        selection = fraction_valid < valid_foreground_threshold
        expt_reflections.set_flags(selection, expt_reflections.flags.dont_integrate)

        expt_reflections["num_pixels.valid"] = sboxs.count_mask_values(MaskCode.Valid)
        expt_reflections["num_pixels.background"] = sboxs.count_mask_values(
            MaskCode.Valid | MaskCode.Background
        )
        expt_reflections["num_pixels.background_used"] = sboxs.count_mask_values(
            MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
        )
        expt_reflections["num_pixels.foreground"] = nvalfg

        logger.info(f"    Calculating background using {params.background_model}")
        params.integration.background.simple.model.algorithm = params.background_model
        background_algorithm = SimpleBackgroundExt(
            params=params, experiments=experiments
        )
        success = background_algorithm.compute_background(expt_reflections)
        expt_reflections.set_flags(
            ~success, expt_reflections.flags.failed_during_background_modelling
        )

        logger.info("    Calculating centroids")
        expt_reflections = compute_tof_centroids(expt, expt_reflections)

        expt_reflections, profile_fitter = integrate_reflection_table_for_experiment(
            expt,
            expt_reflections,
            expt_data,
            params,
            **corrections_data,
        )

        if params.overlaps.estimator == "joint":
            logger.info("    Fitting every reflection with the 3D model")
            expt_reflections = fit_profiles_jointly(
                expt,
                expt_reflections,
                expt_data,
                params,
                profile_fitter,
                coupled_pairs if params.overlaps.mode == "deconvolve" else None,
                **corrections_data,
            )
        elif params.overlaps.mode == "deconvolve":
            logger.info("    Deconvolving overlapped reflections")
            expt_reflections = deconvolve_overlaps(
                expt,
                expt_reflections,
                expt_data,
                params,
                profile_fitter,
                coupled_pairs,
                **corrections_data,
            )

        predicted_reflections.set_selected(sel_expt, expt_reflections)

    integration_report = IntegrationReport(experiments, predicted_reflections)
    logger.info("")
    logger.info(integration_report.as_str(prefix=" "))

    if not params.keep_shoeboxes:
        del predicted_reflections["shoebox"]
    return predicted_reflections


if __name__ == "__main__":
    run()
