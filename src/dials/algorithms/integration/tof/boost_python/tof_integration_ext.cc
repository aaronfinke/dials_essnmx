
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/integration/tof/tof_mask_calculator.h>
#include <dials/algorithms/integration/tof/tof_integration.h>
#include <dials/algorithms/integration/tof/tof_profile_1d_ibix.h>
#include <dials/algorithms/integration/tof/tof_profile_1d_ic.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_gutmann.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_ic.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_ibix.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_lp.h>
#include <dials/algorithms/scaling/tof/tof_scaling.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  std::shared_ptr<ProfileFitter> make_profile_fitter(
    object profile_1d_ibix_params_obj,
    object profile_1d_ic_params_obj,
    object profile_3d_gutmann_params_obj,
    object profile_3d_ic_params_obj,
    object profile_3d_ibix_params_obj,
    object profile_3d_lp_params_obj) {
    // Only one profile fitting method allowed
    int n_given =
      !profile_1d_ibix_params_obj.is_none() + !profile_1d_ic_params_obj.is_none()
      + !profile_3d_gutmann_params_obj.is_none() + !profile_3d_ic_params_obj.is_none()
      + !profile_3d_ibix_params_obj.is_none() + !profile_3d_lp_params_obj.is_none();
    DIALS_ASSERT(n_given <= 1);

    if (!profile_1d_ibix_params_obj.is_none()) {
      return std::make_shared<Profile1DIBIXFitter>(
        extract<TOFProfile1DIBIXParams>(profile_1d_ibix_params_obj));
    }
    if (!profile_1d_ic_params_obj.is_none()) {
      return std::make_shared<Profile1DICFitter>(
        extract<TOFProfile1DICParams>(profile_1d_ic_params_obj));
    }
    if (!profile_3d_gutmann_params_obj.is_none()) {
      return std::make_shared<Profile3DGutmannFitter>(
        extract<TOFProfile3DGutmannParams>(profile_3d_gutmann_params_obj));
    }
    if (!profile_3d_ic_params_obj.is_none()) {
      return std::make_shared<Profile3DICFitter>(
        extract<TOFProfile3DICParams>(profile_3d_ic_params_obj));
    }
    if (!profile_3d_ibix_params_obj.is_none()) {
      return std::make_shared<Profile3DIBIXFitter>(
        extract<TOFProfile3DIBIXParams>(profile_3d_ibix_params_obj));
    }
    if (!profile_3d_lp_params_obj.is_none()) {
      return std::make_shared<Profile3DLPFitter>(
        extract<TOFProfile3DLPParams>(profile_3d_lp_params_obj));
    }
    return nullptr;
  }

  // boost::python::init<> and function-pointer signature deduction only
  // support up to 15 arguments including the implicit self, so
  // TOFProfile1DICParams (15 constructor arguments) and TOFProfile3DICParams
  // (24 constructor arguments) are constructed from a single Python dict
  // instead.
  std::shared_ptr<TOFProfile1DICParams> make_TOFProfile1DICParams(dict params) {
    auto d = [&](const char* key) { return extract<double>(params[key])(); };
    auto i = [&](const char* key) { return extract<int>(params[key])(); };
    auto b = [&](const char* key) { return extract<bool>(params[key])(); };
    return std::make_shared<TOFProfile1DICParams>(d("A"),
                                                  d("A_min"),
                                                  d("A_max"),
                                                  d("B"),
                                                  d("B_min"),
                                                  d("B_max"),
                                                  d("R"),
                                                  d("R_min"),
                                                  d("R_max"),
                                                  d("HatWidth"),
                                                  d("KConv"),
                                                  i("n_restarts"),
                                                  b("optimize_profile"),
                                                  b("show_profile_failures"));
  }

  std::shared_ptr<TOFProfile3DICParams> make_TOFProfile3DICParams(dict params) {
    auto d = [&](const char* key) { return extract<double>(params[key])(); };
    auto i = [&](const char* key) { return extract<int>(params[key])(); };
    auto b = [&](const char* key) { return extract<bool>(params[key])(); };
    return std::make_shared<TOFProfile3DICParams>(d("A"),
                                                  d("A_min"),
                                                  d("A_max"),
                                                  d("B"),
                                                  d("B_min"),
                                                  d("B_max"),
                                                  d("R"),
                                                  d("R_min"),
                                                  d("R_max"),
                                                  d("SigX_min"),
                                                  d("SigX_max"),
                                                  d("SigY_min"),
                                                  d("SigY_max"),
                                                  d("SigP"),
                                                  d("SigP_min"),
                                                  d("SigP_max"),
                                                  d("HatWidth"),
                                                  d("KConv"),
                                                  i("n_restarts"),
                                                  b("optimize_profile"),
                                                  d("max_drift_factor"),
                                                  b("show_profile_failures"));
  }

  std::shared_ptr<TOFProfile3DIBIXParams> make_TOFProfile3DIBIXParams(dict params) {
    auto d = [&](const char* key) { return extract<double>(params[key])(); };
    auto i = [&](const char* key) { return extract<int>(params[key])(); };
    auto b = [&](const char* key) { return extract<bool>(params[key])(); };
    return std::make_shared<TOFProfile3DIBIXParams>(d("alpha"),
                                                    d("alpha_min"),
                                                    d("alpha_max"),
                                                    d("beta"),
                                                    d("beta_min"),
                                                    d("beta_max"),
                                                    d("sigma_min"),
                                                    d("sigma_max"),
                                                    d("SigX_min"),
                                                    d("SigX_max"),
                                                    d("SigY_min"),
                                                    d("SigY_max"),
                                                    d("SigP"),
                                                    d("SigP_min"),
                                                    d("SigP_max"),
                                                    i("n_restarts"),
                                                    b("optimize_profile"),
                                                    d("max_drift_factor"),
                                                    b("show_profile_failures"));
  }

  std::shared_ptr<TOFProfile3DLPParams> make_TOFProfile3DLPParams(dict params) {
    auto d = [&](const char* key) { return extract<double>(params[key])(); };
    auto i = [&](const char* key) { return extract<int>(params[key])(); };
    auto b = [&](const char* key) { return extract<bool>(params[key])(); };
    auto d_or = [&](const char* key, double v) {
      return extract<double>(params.get(key, v))();
    };
    auto b_or = [&](const char* key, bool v) {
      return extract<bool>(params.get(key, v))();
    };
    return std::make_shared<TOFProfile3DLPParams>(
      d("pulse_width"),
      b("optimize_pulse_width"),
      d("pulse_width_min_factor"),
      d("pulse_width_max_factor"),
      d("tau"),
      d("tau_min"),
      d("tau_max"),
      b("optimize_emission"),
      d("sigma"),
      d("sigma_min"),
      d("sigma_max"),
      d("SigX_min"),
      d("SigX_max"),
      d("SigY_min"),
      d("SigY_max"),
      d("SigP"),
      d("SigP_min"),
      d("SigP_max"),
      i("n_restarts"),
      b("optimize_profile"),
      d("max_drift_factor"),
      b("show_profile_failures"),
      d_or("min_correlation", 0.75),
      d_or("plateau_height_fraction", 0.25),
      b_or("library_mode", false),
      d_or("free_position_min_i_sigma", 10.0),
      d_or("position_tolerance", 3.0));
  }

  // Builds the (x, y, z)-ordered 3D arrays the 3D fitters take from flat
  // arrays in c_grid order, index (ix * ny + iy) * nz + iz
  template <typename T>
  scitbx::af::versa<T, af::c_grid<3>> grid_3d(const scitbx::af::const_ref<T>& flat,
                                              std::size_t nx,
                                              std::size_t ny,
                                              std::size_t nz) {
    DIALS_ASSERT(flat.size() == nx * ny * nz);
    scitbx::af::versa<T, af::c_grid<3>> out(af::c_grid<3>(nx, ny, nz));
    for (std::size_t i = 0; i < flat.size(); ++i)
      out[i] = flat[i];
    return out;
  }

  // Fits the long-pulse model to one shoebox given as flat arrays (for
  // testing and diagnostics). Returns (success, I_prf, var_prf, profile)
  boost::python::tuple fit_profile_3d_lp_wrapper(
    scitbx::af::const_ref<vec3<double>> coords,
    scitbx::af::const_ref<double> intensities,
    scitbx::af::const_ref<double> background_variances,
    scitbx::af::const_ref<double> tof_z,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    TOFProfile3DLPParams& params,
    scitbx::af::const_ref<double> gains) {
    scitbx::af::versa<vec3<double>, af::c_grid<3>> c = grid_3d(coords, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> I = grid_3d(intensities, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> B =
      grid_3d(background_variances, nx, ny, nz);
    // No gains given: the intensities are raw counts
    scitbx::af::versa<double, af::c_grid<3>> G =
      gains.size() == 0
        ? scitbx::af::versa<double, af::c_grid<3>>(af::c_grid<3>(nx, ny, nz), 1.0)
        : grid_3d(gains, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> profile(af::c_grid<3>(nx, ny, nz));
    double I_prf = 0.0, var_prf = 0.0;
    bool success = fit_profile_3d_lp(
      c.const_ref(), I, B, G, tof_z, params, I_prf, var_prf, profile, true);
    scitbx::af::shared<double> profile_flat(profile.begin(), profile.end());
    return boost::python::make_tuple(success, I_prf, var_prf, profile_flat);
  }

  // Fits one shoebox, given as flat arrays, with a library shape (for testing
  // and diagnostics). shape holds W, tau, sigma, sig_x, sig_y, sig_p, dxdt,
  // dydt, expected (x, y, t) and scatter (x, y, t). Returns (success, I_prf,
  // var_prf, position_free, fitted centre)
  boost::python::tuple fit_profile_3d_lp_library_wrapper(
    scitbx::af::const_ref<vec3<double>> coords,
    scitbx::af::const_ref<double> intensities,
    scitbx::af::const_ref<double> background_variances,
    scitbx::af::const_ref<double> tof_z,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    const TOFProfile3DLPParams& params,
    dict shape,
    scitbx::af::const_ref<double> gains) {
    auto d = [&](const char* key) { return extract<double>(shape[key])(); };
    auto v = [&](const char* key) { return extract<vec3<double>>(shape[key])(); };
    LPLibraryShape lib;
    lib.W = d("W");
    lib.tau = d("tau");
    lib.sigma = d("sigma");
    lib.SigX = d("sig_x");
    lib.SigY = d("sig_y");
    lib.SigP = d("sig_p");
    lib.dXdt = d("dxdt");
    lib.dYdt = d("dydt");
    const vec3<double> e = v("expected");
    const vec3<double> sc = v("scatter");
    lib.expected_x = e[0];
    lib.expected_y = e[1];
    lib.expected_t = e[2];
    lib.scatter_x = sc[0];
    lib.scatter_y = sc[1];
    lib.scatter_t = sc[2];

    scitbx::af::versa<vec3<double>, af::c_grid<3>> c = grid_3d(coords, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> I = grid_3d(intensities, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> B =
      grid_3d(background_variances, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> G =
      gains.size() == 0
        ? scitbx::af::versa<double, af::c_grid<3>>(af::c_grid<3>(nx, ny, nz), 1.0)
        : grid_3d(gains, nx, ny, nz);
    double I_prf = 0.0, var_prf = 0.0;
    LPFitResult result;
    bool success = fit_profile_3d_lp_library(
      c.const_ref(), I, B, G, tof_z, params, lib, I_prf, var_prf, boost::none, &result);
    return boost::python::make_tuple(
      success,
      I_prf,
      var_prf,
      result.position_free,
      vec3<double>(result.centre_x, result.centre_y, result.centre_t));
  }

  // As fit_profile_3d_lp_wrapper, for the Ikeda-Carpenter model. coords z
  // must be in frames, as integrate_shoebox fills them. Returns
  // (success, I_prf)
  boost::python::tuple fit_profile_3d_ic_wrapper(
    scitbx::af::const_ref<vec3<double>> coords,
    scitbx::af::const_ref<double> intensities,
    scitbx::af::const_ref<double> background_variances,
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    TOFProfile3DICParams& params) {
    scitbx::af::versa<vec3<double>, af::c_grid<3>> c = grid_3d(coords, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> I = grid_3d(intensities, nx, ny, nz);
    scitbx::af::versa<double, af::c_grid<3>> B =
      grid_3d(background_variances, nx, ny, nz);
    double I_prf = 0.0;
    bool success = fit_profile_3d_ic(c.const_ref(), I, B, params, I_prf);
    return boost::python::make_tuple(success, I_prf);
  }

  void integrate_reflection_table_wrapper(dials::af::reflection_table& reflection_table,
                                          dxtbx::model::Experiment& experiment,
                                          dxtbx::ImageSequence& data,
                                          object incident_params_obj,
                                          object absorption_params_obj,
                                          const bool& apply_lorentz,
                                          int n_threads,
                                          object profile_1d_ibix_params_obj,
                                          object profile_1d_ic_params_obj,
                                          object profile_3d_gutmann_params_obj,
                                          object profile_3d_ic_params_obj,
                                          object profile_3d_ibix_params_obj,
                                          object profile_3d_lp_params_obj) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;

    if (!incident_params_obj.is_none()) {
      incident_params =
        extract<dials_scaling::TOFIncidentSpectrumParams>(incident_params_obj);
    }
    if (!absorption_params_obj.is_none()) {
      absorption_params =
        extract<dials_scaling::TOFAbsorptionParams>(absorption_params_obj);
    }

    std::shared_ptr<ProfileFitter> profile_fitter =
      make_profile_fitter(profile_1d_ibix_params_obj,
                          profile_1d_ic_params_obj,
                          profile_3d_gutmann_params_obj,
                          profile_3d_ic_params_obj,
                          profile_3d_ibix_params_obj,
                          profile_3d_lp_params_obj);

    integrate_reflection_table(reflection_table,
                               experiment,
                               data,
                               incident_params,
                               absorption_params,
                               apply_lorentz,
                               n_threads,
                               profile_fitter);
  }

  void extract_correction_params(
    object incident_params_obj,
    object absorption_params_obj,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams>& incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams>& absorption_params) {
    if (!incident_params_obj.is_none()) {
      incident_params =
        extract<dials_scaling::TOFIncidentSpectrumParams>(incident_params_obj);
    }
    if (!absorption_params_obj.is_none()) {
      absorption_params =
        extract<dials_scaling::TOFAbsorptionParams>(absorption_params_obj);
    }
  }

  boost::python::tuple calculate_line_profile_for_reflection_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection(reflection,
                                                 experiment,
                                                 data,
                                                 incident_params,
                                                 absorption_params,
                                                 raw_projected_intensity_out,
                                                 projected_intensity_out,
                                                 projected_background_out,
                                                 tof_z_out,
                                                 apply_lorentz_correction);
  }

  boost::python::tuple calculate_line_profile_for_reflection_1d_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    scitbx::af::shared<double> line_profile_out,
    const bool& apply_lorentz_correction,
    TOFProfile1DIBIXParams& profile_params_1d_ibix) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection(reflection,
                                                 experiment,
                                                 data,
                                                 incident_params,
                                                 absorption_params,
                                                 raw_projected_intensity_out,
                                                 projected_intensity_out,
                                                 projected_background_out,
                                                 tof_z_out,
                                                 line_profile_out,
                                                 apply_lorentz_correction,
                                                 profile_params_1d_ibix);
  }

  boost::python::tuple calculate_line_profile_for_reflection_1d_ic_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    scitbx::af::shared<double> line_profile_out,
    const bool& apply_lorentz_correction,
    TOFProfile1DICParams& profile_params_1d_ic) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection(reflection,
                                                 experiment,
                                                 data,
                                                 incident_params,
                                                 absorption_params,
                                                 raw_projected_intensity_out,
                                                 projected_intensity_out,
                                                 projected_background_out,
                                                 tof_z_out,
                                                 line_profile_out,
                                                 apply_lorentz_correction,
                                                 profile_params_1d_ic);
  }

  boost::python::tuple calculate_line_profile_for_reflection_3d_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DGutmannParams& profile_params_3d_gutmann) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection_3d(reflection,
                                                    experiment,
                                                    data,
                                                    incident_params,
                                                    absorption_params,
                                                    raw_projected_intensity_out,
                                                    projected_intensity_out,
                                                    projected_background_out,
                                                    tof_z_out,
                                                    apply_lorentz_correction,
                                                    profile_params_3d_gutmann);
  }

  boost::python::tuple calculate_line_profile_for_reflection_3d_ic_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DICParams& profile_params_3d_ic) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection_3d(reflection,
                                                    experiment,
                                                    data,
                                                    incident_params,
                                                    absorption_params,
                                                    raw_projected_intensity_out,
                                                    projected_intensity_out,
                                                    projected_background_out,
                                                    tof_z_out,
                                                    apply_lorentz_correction,
                                                    profile_params_3d_ic);
  }

  boost::python::tuple calculate_line_profile_for_reflection_3d_ibix_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DIBIXParams& profile_params_3d_ibix) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection_3d(reflection,
                                                    experiment,
                                                    data,
                                                    incident_params,
                                                    absorption_params,
                                                    raw_projected_intensity_out,
                                                    projected_intensity_out,
                                                    projected_background_out,
                                                    tof_z_out,
                                                    apply_lorentz_correction,
                                                    profile_params_3d_ibix);
  }

  boost::python::tuple calculate_line_profile_for_reflection_3d_lp_wrapper(
    dials::af::reflection_table& reflection,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DLPParams& profile_params_3d_lp) {
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params;
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params;
    extract_correction_params(
      incident_params_obj, absorption_params_obj, incident_params, absorption_params);

    return calculate_line_profile_for_reflection_3d(reflection,
                                                    experiment,
                                                    data,
                                                    incident_params,
                                                    absorption_params,
                                                    raw_projected_intensity_out,
                                                    projected_intensity_out,
                                                    projected_background_out,
                                                    tof_z_out,
                                                    apply_lorentz_correction,
                                                    profile_params_3d_lp);
  }

  BOOST_PYTHON_MODULE(dials_algorithms_tof_integration_ext) {
    class_<TOFProfile1DIBIXParams>("TOFProfile1DIBIXParams", no_init)
      .def(init<double,
                double,
                double,
                double,
                double,
                double,
                double,
                double,
                int,
                bool,
                bool>())
      .def_readwrite("A", &TOFProfile1DIBIXParams::A)
      .def_readwrite("A_min", &TOFProfile1DIBIXParams::A_min)
      .def_readwrite("A_max", &TOFProfile1DIBIXParams::A_max)
      .def_readwrite("alpha", &TOFProfile1DIBIXParams::alpha)
      .def_readwrite("alpha_min", &TOFProfile1DIBIXParams::alpha_min)
      .def_readwrite("alpha_max", &TOFProfile1DIBIXParams::alpha_max)
      .def_readwrite("beta", &TOFProfile1DIBIXParams::beta)
      .def_readwrite("beta_min", &TOFProfile1DIBIXParams::beta_min)
      .def_readwrite("beta_max", &TOFProfile1DIBIXParams::beta_max)
      .def_readwrite("n_restarts", &TOFProfile1DIBIXParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile1DIBIXParams::optimize_profile)
      .def_readwrite("show_profile_failures",
                     &TOFProfile1DIBIXParams::show_profile_failures);

    class_<TOFProfile1DICParams, std::shared_ptr<TOFProfile1DICParams>>(
      "TOFProfile1DICParams", no_init)
      .def("__init__", make_constructor(&make_TOFProfile1DICParams))
      .def_readwrite("A", &TOFProfile1DICParams::A)
      .def_readwrite("A_min", &TOFProfile1DICParams::A_min)
      .def_readwrite("A_max", &TOFProfile1DICParams::A_max)
      .def_readwrite("B", &TOFProfile1DICParams::B)
      .def_readwrite("B_min", &TOFProfile1DICParams::B_min)
      .def_readwrite("B_max", &TOFProfile1DICParams::B_max)
      .def_readwrite("R", &TOFProfile1DICParams::R)
      .def_readwrite("R_min", &TOFProfile1DICParams::R_min)
      .def_readwrite("R_max", &TOFProfile1DICParams::R_max)
      .def_readwrite("HatWidth", &TOFProfile1DICParams::HatWidth)
      .def_readwrite("KConv", &TOFProfile1DICParams::KConv)
      .def_readwrite("n_restarts", &TOFProfile1DICParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile1DICParams::optimize_profile)
      .def_readwrite("show_profile_failures",
                     &TOFProfile1DICParams::show_profile_failures);

    class_<TOFProfile3DGutmannParams>("TOFProfile3DGutmannParams", no_init)
      .def(
        init<double, double, double, double, double, double, int, bool, bool, bool>())
      .def_readwrite("alpha", &TOFProfile3DGutmannParams::alpha)
      .def_readwrite("alpha_min", &TOFProfile3DGutmannParams::alpha_min)
      .def_readwrite("alpha_max", &TOFProfile3DGutmannParams::alpha_max)
      .def_readwrite("beta", &TOFProfile3DGutmannParams::beta)
      .def_readwrite("beta_min", &TOFProfile3DGutmannParams::beta_min)
      .def_readwrite("beta_max", &TOFProfile3DGutmannParams::beta_max)
      .def_readwrite("n_restarts", &TOFProfile3DGutmannParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile3DGutmannParams::optimize_profile)
      .def_readwrite("use_central_diff", &TOFProfile3DGutmannParams::use_central_diff)
      .def_readwrite("show_profile_failures",
                     &TOFProfile3DGutmannParams::show_profile_failures);

    class_<TOFProfile3DICParams, std::shared_ptr<TOFProfile3DICParams>>(
      "TOFProfile3DICParams", no_init)
      .def("__init__", make_constructor(&make_TOFProfile3DICParams))
      .def_readwrite("A", &TOFProfile3DICParams::A)
      .def_readwrite("A_min", &TOFProfile3DICParams::A_min)
      .def_readwrite("A_max", &TOFProfile3DICParams::A_max)
      .def_readwrite("B", &TOFProfile3DICParams::B)
      .def_readwrite("B_min", &TOFProfile3DICParams::B_min)
      .def_readwrite("B_max", &TOFProfile3DICParams::B_max)
      .def_readwrite("R", &TOFProfile3DICParams::R)
      .def_readwrite("R_min", &TOFProfile3DICParams::R_min)
      .def_readwrite("R_max", &TOFProfile3DICParams::R_max)
      .def_readwrite("SigX_min", &TOFProfile3DICParams::SigX_min)
      .def_readwrite("SigX_max", &TOFProfile3DICParams::SigX_max)
      .def_readwrite("SigY_min", &TOFProfile3DICParams::SigY_min)
      .def_readwrite("SigY_max", &TOFProfile3DICParams::SigY_max)
      .def_readwrite("SigP", &TOFProfile3DICParams::SigP)
      .def_readwrite("SigP_min", &TOFProfile3DICParams::SigP_min)
      .def_readwrite("SigP_max", &TOFProfile3DICParams::SigP_max)
      .def_readwrite("HatWidth", &TOFProfile3DICParams::HatWidth)
      .def_readwrite("KConv", &TOFProfile3DICParams::KConv)
      .def_readwrite("n_restarts", &TOFProfile3DICParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile3DICParams::optimize_profile)
      .def_readwrite("max_drift_factor", &TOFProfile3DICParams::max_drift_factor)
      .def_readwrite("show_profile_failures",
                     &TOFProfile3DICParams::show_profile_failures);

    class_<TOFProfile3DIBIXParams, std::shared_ptr<TOFProfile3DIBIXParams>>(
      "TOFProfile3DIBIXParams", no_init)
      .def("__init__", make_constructor(&make_TOFProfile3DIBIXParams))
      .def_readwrite("alpha", &TOFProfile3DIBIXParams::alpha)
      .def_readwrite("alpha_min", &TOFProfile3DIBIXParams::alpha_min)
      .def_readwrite("alpha_max", &TOFProfile3DIBIXParams::alpha_max)
      .def_readwrite("beta", &TOFProfile3DIBIXParams::beta)
      .def_readwrite("beta_min", &TOFProfile3DIBIXParams::beta_min)
      .def_readwrite("beta_max", &TOFProfile3DIBIXParams::beta_max)
      .def_readwrite("sigma_min", &TOFProfile3DIBIXParams::sigma_min)
      .def_readwrite("sigma_max", &TOFProfile3DIBIXParams::sigma_max)
      .def_readwrite("SigX_min", &TOFProfile3DIBIXParams::SigX_min)
      .def_readwrite("SigX_max", &TOFProfile3DIBIXParams::SigX_max)
      .def_readwrite("SigY_min", &TOFProfile3DIBIXParams::SigY_min)
      .def_readwrite("SigY_max", &TOFProfile3DIBIXParams::SigY_max)
      .def_readwrite("SigP", &TOFProfile3DIBIXParams::SigP)
      .def_readwrite("SigP_min", &TOFProfile3DIBIXParams::SigP_min)
      .def_readwrite("SigP_max", &TOFProfile3DIBIXParams::SigP_max)
      .def_readwrite("n_restarts", &TOFProfile3DIBIXParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile3DIBIXParams::optimize_profile)
      .def_readwrite("max_drift_factor", &TOFProfile3DIBIXParams::max_drift_factor)
      .def_readwrite("show_profile_failures",
                     &TOFProfile3DIBIXParams::show_profile_failures);

    class_<TOFProfile3DLPParams, std::shared_ptr<TOFProfile3DLPParams>>(
      "TOFProfile3DLPParams", no_init)
      .def("__init__", make_constructor(&make_TOFProfile3DLPParams))
      .def_readwrite("pulse_width", &TOFProfile3DLPParams::pulse_width)
      .def_readwrite("optimize_pulse_width",
                     &TOFProfile3DLPParams::optimize_pulse_width)
      .def_readwrite("pulse_width_min_factor",
                     &TOFProfile3DLPParams::pulse_width_min_factor)
      .def_readwrite("pulse_width_max_factor",
                     &TOFProfile3DLPParams::pulse_width_max_factor)
      .def_readwrite("tau", &TOFProfile3DLPParams::tau)
      .def_readwrite("tau_min", &TOFProfile3DLPParams::tau_min)
      .def_readwrite("tau_max", &TOFProfile3DLPParams::tau_max)
      .def_readwrite("optimize_emission", &TOFProfile3DLPParams::optimize_emission)
      .def_readwrite("sigma", &TOFProfile3DLPParams::sigma)
      .def_readwrite("sigma_min", &TOFProfile3DLPParams::sigma_min)
      .def_readwrite("sigma_max", &TOFProfile3DLPParams::sigma_max)
      .def_readwrite("SigX_min", &TOFProfile3DLPParams::SigX_min)
      .def_readwrite("SigX_max", &TOFProfile3DLPParams::SigX_max)
      .def_readwrite("SigY_min", &TOFProfile3DLPParams::SigY_min)
      .def_readwrite("SigY_max", &TOFProfile3DLPParams::SigY_max)
      .def_readwrite("SigP", &TOFProfile3DLPParams::SigP)
      .def_readwrite("SigP_min", &TOFProfile3DLPParams::SigP_min)
      .def_readwrite("SigP_max", &TOFProfile3DLPParams::SigP_max)
      .def_readwrite("n_restarts", &TOFProfile3DLPParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile3DLPParams::optimize_profile)
      .def_readwrite("max_drift_factor", &TOFProfile3DLPParams::max_drift_factor)
      .def_readwrite("show_profile_failures",
                     &TOFProfile3DLPParams::show_profile_failures)
      .def_readwrite("min_correlation", &TOFProfile3DLPParams::min_correlation)
      .def_readwrite("plateau_height_fraction",
                     &TOFProfile3DLPParams::plateau_height_fraction)
      .def_readwrite("library_mode", &TOFProfile3DLPParams::library_mode)
      .def_readwrite("free_position_min_i_sigma",
                     &TOFProfile3DLPParams::free_position_min_i_sigma)
      .def_readwrite("position_tolerance", &TOFProfile3DLPParams::position_tolerance);

    def("tof_long_pulse_line_shape",
        &lp_raw,
        (arg("tof"), arg("T0"), arg("W"), arg("tau"), arg("sigma")));

    def("tof_emg_cdf", &lp_emg_cdf, (arg("t"), arg("tau"), arg("sigma")));

    def("fit_profile_3d_lp",
        &fit_profile_3d_lp_wrapper,
        (arg("coords"),
         arg("intensities"),
         arg("background_variances"),
         arg("tof_z"),
         arg("nx"),
         arg("ny"),
         arg("nz"),
         arg("params"),
         arg("gains") = scitbx::af::shared<double>()));

    def("fit_profile_3d_lp_library",
        &fit_profile_3d_lp_library_wrapper,
        (arg("coords"),
         arg("intensities"),
         arg("background_variances"),
         arg("tof_z"),
         arg("nx"),
         arg("ny"),
         arg("nz"),
         arg("params"),
         arg("shape"),
         arg("gains") = scitbx::af::shared<double>()));

    def("fit_profile_3d_ic",
        &fit_profile_3d_ic_wrapper,
        (arg("coords"),
         arg("intensities"),
         arg("background_variances"),
         arg("nx"),
         arg("ny"),
         arg("nz"),
         arg("params")));

    def("tof_calculate_ellipse_shoebox_mask",
        &tof_calculate_ellipse_shoebox_mask,
        (arg("reflection_table"),
         arg("experiment"),
         arg("n_threads") = 1,
         arg("scale") = 1));

    def("tof_calculate_seed_skewness_shoebox_mask",
        &tof_calculate_seed_skewness_shoebox_mask,
        (arg("reflection_table"),
         arg("experiment"),
         arg("d_skewness_threshold"),
         arg("min_iterations"),
         arg("n_threads") = 1));

    def("integrate_reflection_table",
        &integrate_reflection_table_wrapper,
        (arg("reflection_table"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("apply_lorentz_correction"),
         arg("n_threads"),
         arg("profile_1d_ibix_params") = object(),
         arg("profile_1d_ic_params") = object(),
         arg("profile_3d_gutmann_params") = object(),
         arg("profile_3d_ic_params") = object(),
         arg("profile_3d_ibix_params") = object(),
         arg("profile_3d_lp_params") = object()));

    def("calculate_line_profile_for_reflection",
        &calculate_line_profile_for_reflection_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params") = object(),
         arg("absorption_params") = object(),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("apply_lorentz_correction")));

    def("calculate_line_profile_for_reflection",
        &calculate_line_profile_for_reflection_1d_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("line_profile_out"),
         arg("apply_lorentz_correction"),
         arg("profile_params_1d_ibix")));

    def("calculate_line_profile_for_reflection",
        &calculate_line_profile_for_reflection_1d_ic_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("line_profile_out"),
         arg("apply_lorentz_correction"),
         arg("profile_params_1d_ic")));

    def("calculate_line_profile_for_reflection_3d",
        &calculate_line_profile_for_reflection_3d_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("apply_lorentz_correction"),
         arg("profile_params_3d_gutmann")));

    def("calculate_line_profile_for_reflection_3d",
        &calculate_line_profile_for_reflection_3d_ic_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("apply_lorentz_correction"),
         arg("profile_params_3d_ic")));

    def("calculate_line_profile_for_reflection_3d",
        &calculate_line_profile_for_reflection_3d_ibix_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("apply_lorentz_correction"),
         arg("profile_params_3d_ibix")));

    def("calculate_line_profile_for_reflection_3d",
        &calculate_line_profile_for_reflection_3d_lp_wrapper,
        (arg("reflection"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("raw_projected_intensity_out"),
         arg("projected_intensity_out"),
         arg("projected_background_out"),
         arg("tof_z_out"),
         arg("apply_lorentz_correction"),
         arg("profile_params_3d_lp")));
  }

}}}  // namespace dials::algorithms::boost_python
