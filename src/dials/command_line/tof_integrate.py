# LIBTBX_SET_DISPATCHER_NAME dials.tof_integrate
from __future__ import annotations

import logging
import multiprocessing
import re
import sys
import warnings
from copy import deepcopy
from math import ceil, floor
from typing import Dict, Tuple

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
    TOFProfile3DGutmannParams,
    TOFProfile3DIBIXParams,
    TOFProfile3DICParams,
    TOFProfile3DLPParams,
    integrate_reflection_table,
    tof_calculate_ellipse_shoebox_mask,
    tof_calculate_seed_skewness_shoebox_mask,
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
method = *summation profile_1d_ibix profile_1d_ic profile_3d_gutmann profile_3d_ic profile_3d_ibix profile_3d_lp
    .type = choice
    .help = "Integration method: "
            "summation: shoebox summation"
            "profile_1d_ibix: https://doi.org/10.1038/srep36628 "
            "profile_1d_ic: Ikeda-Carpenter model, "
            "https://doi.org/10.1016/0168-9002(85)90033-6"
            "profile_3d_gutmann: https://doi.org/10.1016/j.nima.2016.12.026"
            "profile_3d_ic: separable Ikeda-Carpenter x bivariate Gaussian model, "
            "https://doi.org/10.1016/0168-9002(85)90033-6"
            "profile_3d_ibix: separable iBIX x bivariate Gaussian model, "
            "https://doi.org/10.1038/srep36628 "
            "profile_3d_lp: separable long-pulse x bivariate Gaussian model for "
            "long-pulse sources (e.g. ESS NMX): rectangular proton pulse "
            "convolved with an exponential moderator emission and a Gaussian"

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
    absorption{
        incident_spectrum{
            sample_number_density = 0.0722
                .type = float
                .help = "Sample number density for incident run."
                        "Default is Vanadium used at SXD"
            sample_radius = 3.0
                .type = float
                .help = "Sample radius (mm) for incident run."
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
profile_3d_lp{
    pulse_width = 2860.0
        .type = float(value_min=0)
        .help = "Proton pulse length (us): the width of the plateau of a peak "
                "in time of flight. 2860 us at ESS."
    optimize_pulse_width = False
        .type = bool
        .help = "Refine the plateau width per reflection, within "
                "[min_pulse_width_factor, max_pulse_width_factor] x pulse_width. "
                "If False it is held at pulse_width."
    min_pulse_width_factor = 0.5
        .type = float(value_min=0)
        .help = "Lower bound of the plateau width, as a fraction of pulse_width"
    max_pulse_width_factor = 1.5
        .type = float(value_min=0)
        .help = "Upper bound of the plateau width, as a fraction of pulse_width"
    tau = 100.0
        .type = float(value_min=0)
        .help = "Moderator emission time constant (us). It is degenerate with "
                "the onset and sigma on the leading edge of the peak, so it is "
                "best calibrated once and held fixed."
    optimize_emission = False
        .type = bool
        .help = "Refine tau per reflection, within [min_tau, max_tau]"
    min_tau = 1.0
        .type = float(value_min=0)
        .help = "Min moderator emission time constant (us) for optimization"
    max_tau = 2000.0
        .type = float(value_min=0)
        .help = "Max moderator emission time constant (us) for optimization"
    init_sigma = 300.0
        .type = float(value_min=0)
        .help = "Initial Gaussian broadening (us): mosaicity, divergence, "
                "detector and chopper timing"
    min_sigma = 1.0
        .type = float(value_min=0)
        .help = "Min Gaussian broadening (us) for optimization"
    max_sigma = 5000.0
        .type = float(value_min=0)
        .help = "Max Gaussian broadening (us) for optimization"
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
    min_sig_p = -0.95
        .type = float
        .help = "Min bivariate Gaussian spatial correlation for optimization"
    max_sig_p = 0.95
        .type = float
        .help = "Max bivariate Gaussian spatial correlation for optimization"
    n_restarts = 5
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params. "
                "Each failed attempt costs about as much as a full fit, so this "
                "mostly sets the time spent on reflections that are rejected."
    max_drift_factor = 1.5
        .type = float
        .help = "Scales the per-reflection dXdt/dYdt bounds relative to the "
                "shoebox's spatial extent divided by its ToF extent"
    min_correlation = 0.75
        .type = float(value_min=0, value_max=1)
        .help = "Correlation with the data that a reflection fitted on its own must"
                "reach, both for its ToF projection (over the whole box) and for its"
                "XY projection (over the pixels where the model is at least 5% of"
                "its maximum)."
    plateau_height_fraction = 0.25
        .type = float(value_min=0)
        .help = "Allowed relative difference between the plateau height of the"
                "ToF projection of a reflection fitted on its own and the data's"
                "mean over the plateau. 0 disables the test."
    library{
        build = False
            .type = bool
            .help = "Measure the shape on strong reflections, and fit every"
                    "reflection with the shape of the nearest of those held"
                    "fixed, at its prediction corrected by the median offset the"
                    "library measured on its panel. Needs predictions (xyzcal.px"
                    "and tof_cal)."
        min_i_sigma = 20.0
            .type = float(value_min=0)
            .help = "Summation I/sigma a reflection must reach for its fitted"
                    "shape to enter the library."
        min_correlation = 0.85
            .type = float(value_min=0, value_max=1)
            .help = "Correlation of both projections a free fit must reach for"
                    "its shape to enter the library."
        min_entries_per_panel = 10
            .type = int(value_min=1)
            .help = "Entries a panel must hold for its own median offset from the"
                    "predictions to be used; otherwise the whole detector's is."
        free_position_min_i_sigma = 10.0
            .type = float
            .help = "A reflection is first fitted at its expected position with"
                    "only the amplitude free and the shape broadened by how far"
                    "the library's own peaks scatter about their expected"
                    "positions: a weak peak's counts cannot place it, and a"
                    "position fitted to them finds the brightest noise and reads"
                    "the intensity high. Where that first fit reaches this"
                    "I/sigma the position is fitted as well, with the"
                    "unbroadened shape."
        position_tolerance = 3.0
            .type = float(value_min=0)
            .help = "How far a fitted position may move from the expected one:"
                    "this many times the scatter of the library's own offsets"
                    "(at least 1 px on the detector and one ToF bin)."
    }
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


def integrate_reflection_table_for_experiment(
    expt: Experiment,
    expt_reflections: flex.reflection_table,
    expt_data: ImageSequence,
    params: libtbx.phil.scope_extract,
    **kwargs: Dict,
) -> flex.reflection_table:
    apply_lorentz = params.corrections.lorentz
    profile_1d_ibix_params = None
    profile_1d_ic_params = None
    profile_3d_gutmann_params = None
    profile_3d_ic_params = None
    profile_3d_ibix_params = None
    profile_3d_lp_params = None
    incident_params = None
    absorption_params = None

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

    elif params.method == "profile_3d_lp":
        p = params.profile_3d_lp
        if p.min_pulse_width_factor > p.max_pulse_width_factor:
            raise ValueError(
                f"profile_3d_lp.min_pulse_width_factor ({p.min_pulse_width_factor}) "
                f"is larger than profile_3d_lp.max_pulse_width_factor "
                f"({p.max_pulse_width_factor})"
            )
        for name in ("tau", "sigma"):
            lo, hi = getattr(p, f"min_{name}"), getattr(p, f"max_{name}")
            if lo > hi:
                raise ValueError(
                    f"profile_3d_lp.min_{name} ({lo}) is larger than "
                    f"profile_3d_lp.max_{name} ({hi})"
                )
        # dict needed due to boost limit of 15 template arguments
        profile_3d_lp_params = TOFProfile3DLPParams(
            {
                "pulse_width": p.pulse_width,
                "optimize_pulse_width": p.optimize_pulse_width,
                "pulse_width_min_factor": p.min_pulse_width_factor,
                "pulse_width_max_factor": p.max_pulse_width_factor,
                "tau": p.tau,
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
                "show_profile_failures": show_profile_failures,
                "min_correlation": p.min_correlation,
                "plateau_height_fraction": p.plateau_height_fraction,
                "library_mode": False,
                "free_position_min_i_sigma": p.library.free_position_min_i_sigma,
                "position_tolerance": p.library.position_tolerance,
            }
        )

    if apply_lorentz:
        logger.info("    Adding Lorentz correction")

    if "incident_data" in kwargs:
        incident_params = TOFIncidentSpectrumParams(
            kwargs["incident_data"],
            kwargs["empty_data"],
            kwargs["expt_proton_charge"],
            kwargs["incident_proton_charge"],
            kwargs["empty_proton_charge"],
        )
        logger.info("    Adding incident spectrum correction")

        if "absorption_params" in kwargs:
            logger.info("    Adding absorption correction")
            absorption_params = kwargs["absorption_params"]

    integrate_reflection_table(
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
        profile_3d_lp_params,
    )

    if profile_3d_lp_params is not None and params.profile_3d_lp.library.build:
        if set_lp_library_columns(expt_reflections, params.profile_3d_lp.library):
            profile_3d_lp_params.library_mode = True
            logger.info("    Fitting every reflection with its library shape")
            integrate_reflection_table(
                expt_reflections,
                expt,
                expt_data,
                incident_params,
                absorption_params,
                apply_lorentz,
                params.mp.nproc,
                profile_3d_lp_params=profile_3d_lp_params,
            )
        for name in list(expt_reflections.keys()):
            if name.startswith("lp_library."):
                del expt_reflections[name]

    return expt_reflections


def _robust_sd(values: np.ndarray) -> float:
    """1.4826 times the median absolute deviation from the median"""
    if len(values) == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def set_lp_library_columns(
    reflections: flex.reflection_table, library_params: libtbx.phil.scope_extract
) -> bool:
    """
    Build the profile_3d_lp library from a first, free pass and write, for
    every reflection, the shape of the nearest library entry (same panel where
    it holds any) and where the reflection is expected: its prediction
    corrected by the median offset of the library's own fitted centres from
    their predictions, on its panel where that holds min_entries_per_panel
    entries and over the whole detector otherwise. The scatter of the entries
    about those medians says how well a position is known. Returns False if
    no library could be built.
    """

    n = len(reflections)
    if n == 0 or "tof_cal" not in reflections or "xyzcal.px" not in reflections:
        logger.info("    No predictions (xyzcal.px, tof_cal): no profile library")
        return False

    shape_names = ("W", "tau", "sigma", "sig_x", "sig_y", "sig_p", "dxdt", "dydt")
    shape = np.column_stack(
        [reflections[f"profile.lp.{k}"].as_numpy_array() for k in shape_names]
    )
    centre = reflections["profile.lp.centre"].as_numpy_array()
    cc = np.minimum(
        reflections["profile.lp.cc_tof"].as_numpy_array(),
        reflections["profile.lp.cc_xy"].as_numpy_array(),
    )
    pred = np.column_stack(
        [
            reflections["xyzcal.px"].parts()[0].as_numpy_array(),
            reflections["xyzcal.px"].parts()[1].as_numpy_array(),
            reflections["tof_cal"].as_numpy_array(),
        ]
    )
    panel = reflections["panel"].as_numpy_array().astype(int)
    I = reflections["intensity.sum.value"].as_numpy_array()
    V = reflections["intensity.sum.variance"].as_numpy_array()
    i_sigma = np.where(V > 0, I / np.sqrt(np.where(V > 0, V, 1.0)), 0.0)
    fitted = reflections.get_flags(reflections.flags.integrated_prf).as_numpy_array()

    entry = (
        fitted
        & (i_sigma >= library_params.min_i_sigma)
        & (cc >= library_params.min_correlation)
        & np.all(np.isfinite(shape), axis=1)
        & np.all(np.isfinite(centre), axis=1)
        & np.all(np.isfinite(pred), axis=1)
    )
    n_entries = int(entry.sum())
    logger.info(
        f"    Profile library holds {n_entries} shapes "
        f"(summation I/sigma >= {library_params.min_i_sigma}, "
        f"correlation >= {library_params.min_correlation})"
    )
    if n_entries == 0:
        return False

    # Offsets of the library's fitted centres from their predictions
    offsets = centre[entry] - pred[entry]
    entry_panel = panel[entry]
    global_offset = np.median(offsets, axis=0)
    panel_offset = {}
    for p in np.unique(entry_panel):
        sel = entry_panel == p
        if sel.sum() >= library_params.min_entries_per_panel:
            panel_offset[int(p)] = np.median(offsets[sel], axis=0)
    applied = np.array(
        [panel_offset.get(int(p), global_offset) for p in entry_panel]
    ).reshape(-1, 3)
    scatter = np.array([_robust_sd(offsets[:, k] - applied[:, k]) for k in range(3)])
    logger.info(
        "    Library offset from the predictions (x px, y px, ToF us): "
        + ", ".join(f"{v:.2f}" for v in global_offset)
        + "; scatter "
        + ", ".join(f"{v:.2f}" for v in scatter)
        + f"; panels with their own offset: {sorted(panel_offset)}"
    )

    # Nearest entry on the same panel (any panel where it has none)
    entry_idx = np.where(entry)[0]
    nearest = np.empty(n, dtype=int)
    for p in np.unique(panel):
        rows = np.where(panel == p)[0]
        candidates = entry_idx[panel[entry_idx] == p]
        if len(candidates) == 0:
            candidates = entry_idx
        d2 = (pred[rows, None, 0] - pred[None, candidates, 0]) ** 2 + (
            pred[rows, None, 1] - pred[None, candidates, 1]
        ) ** 2
        nearest[rows] = candidates[np.argmin(d2, axis=1)]

    expected = pred + np.array(
        [panel_offset.get(int(p), global_offset) for p in panel]
    ).reshape(-1, 3)
    valid = np.all(np.isfinite(pred), axis=1)
    for k, name in enumerate(shape_names):
        reflections[f"lp_library.{name}"] = flex.double(
            np.ascontiguousarray(shape[nearest, k])
        )

    def vec3(a):
        return flex.vec3_double(
            *(flex.double(np.ascontiguousarray(a[:, k])) for k in range(3))
        )

    reflections["lp_library.expected"] = vec3(np.nan_to_num(expected))
    reflections["lp_library.scatter"] = vec3(np.tile(scatter, (n, 1)))
    reflections["lp_library.valid"] = flex.bool(valid.tolist())
    return True


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

    predicted_reflections = remove_overlapping_reflections(predicted_reflections)
    predicted_reflections = remove_out_of_range_reflections(
        predicted_reflections, params
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

        expt_reflections = calculate_shoebox_masks(expt, expt_reflections, params)
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

        expt_reflections = integrate_reflection_table_for_experiment(
            expt,
            expt_reflections,
            expt_data,
            params,
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
