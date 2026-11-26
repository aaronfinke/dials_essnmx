
#include <boost/python.hpp>
#include <string>
#include <boost/python/def.hpp>
#include <dials/algorithms/integration/tof/tof_mask_calculator.h>
#include <dials/algorithms/integration/tof/tof_integration.h>
#include <dials/algorithms/integration/tof/tof_overlaps.h>
#include <dials/algorithms/integration/tof/tof_overlap_deconvolution.h>
#include <dials/algorithms/integration/tof/tof_profile_1d.h>
#include <dials/algorithms/integration/tof/tof_profile_3d.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_lp.h>
#include <dials/algorithms/scaling/tof/tof_scaling.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  std::shared_ptr<ProfileFitter> make_profile_fitter(object profile_1d_params_obj,
                                                     object profile_3d_params_obj,
                                                     object profile_3d_lp_params_obj) {
    // Only one profile fitting method allowed
    int n_given = !profile_1d_params_obj.is_none() + !profile_3d_params_obj.is_none()
                  + !profile_3d_lp_params_obj.is_none();
    DIALS_ASSERT(n_given <= 1);

    if (!profile_1d_params_obj.is_none()) {
      return std::make_shared<Profile1DFitter>(
        extract<TOFProfile1DParams>(profile_1d_params_obj)());
    }
    if (!profile_3d_params_obj.is_none()) {
      return std::make_shared<Profile3DFitter>(
        extract<TOFProfile3DParams>(profile_3d_params_obj)());
    }
    if (!profile_3d_lp_params_obj.is_none()) {
      return std::make_shared<Profile3DLPFitter>(
        extract<TOFProfile3DLPParams>(profile_3d_lp_params_obj));
    }
    return nullptr;
  }

  // boost::python::init<> and function-pointer signature deduction only
  // support up to 15 arguments including the implicit self, so parameter
  // structs with more constructor arguments are constructed from a single
  // Python dict instead.

  // An optional line shape in a parameter dict: "line_kind" ("ibix" or "ic")
  // with "line_init", "line_min" and "line_max", five numbers each
  template <typename Params>
  void read_line_shape(dict params, Params& p) {
    if (!params.has_key("line_kind")) {
      return;
    }
    std::string kind = extract<std::string>(params["line_kind"]);
    DIALS_ASSERT(kind == "ibix" || kind == "ic" || kind == "gutmann");
    p.line_kind = kind == "ic"        ? LineShape::IC
                  : kind == "gutmann" ? LineShape::GUTMANN
                                      : LineShape::IBIX;
    const char* keys[3] = {"line_init", "line_min", "line_max"};
    std::array<double, 5>* arrays[3] = {&p.line_init, &p.line_min, &p.line_max};
    for (int k = 0; k < 3; ++k) {
      boost::python::list values = extract<boost::python::list>(params[keys[k]]);
      DIALS_ASSERT(boost::python::len(values) == 5);
      for (int j = 0; j < 5; ++j) {
        (*arrays[k])[j] = extract<double>(values[j]);
      }
    }
  }

  std::shared_ptr<TOFProfile1DParams> make_TOFProfile1DParams(dict params) {
    auto d = [&](const char* key) { return extract<double>(params[key])(); };
    auto i = [&](const char* key) { return extract<int>(params[key])(); };
    auto b = [&](const char* key) { return extract<bool>(params[key])(); };
    auto p = std::make_shared<TOFProfile1DParams>(d("A_min"),
                                                  d("A_max"),
                                                  d("alpha"),
                                                  d("alpha_min"),
                                                  d("alpha_max"),
                                                  d("beta"),
                                                  d("beta_min"),
                                                  d("beta_max"),
                                                  i("n_restarts"),
                                                  b("optimize_profile"),
                                                  b("show_profile_failures"),
                                                  d("peak_height_smoothing"),
                                                  d("trust_min_corr"),
                                                  d("trust_peak_tolerance"),
                                                  d("trust_peak_height_fraction"),
                                                  d("library_min_i_sigma"),
                                                  d("library_min_corr"));
    if (params.has_key("build_library")) {
      p->build_library = b("build_library");
    }
    read_line_shape(params, *p);
    return p;
  }

  std::shared_ptr<TOFProfile3DParams> make_TOFProfile3DParams(dict params) {
    auto d = [&](const char* key) { return extract<double>(params[key])(); };
    auto i = [&](const char* key) { return extract<int>(params[key])(); };
    auto b = [&](const char* key) { return extract<bool>(params[key])(); };
    auto p = std::make_shared<TOFProfile3DParams>(d("alpha"),
                                                  d("alpha_min"),
                                                  d("alpha_max"),
                                                  d("beta"),
                                                  d("beta_min"),
                                                  d("beta_max"),
                                                  d("SigX_min"),
                                                  d("SigX_max"),
                                                  d("SigY_min"),
                                                  d("SigY_max"),
                                                  d("SigP_min"),
                                                  d("SigP_max"),
                                                  d("max_drift_factor"),
                                                  i("n_restarts"),
                                                  b("show_profile_failures"),
                                                  d("library_min_i_sigma"),
                                                  d("library_min_corr"),
                                                  d("position_tolerance"),
                                                  d("free_position_min_i_sigma"));
    // Without its own value a free fit is held to the library's threshold
    p->min_corr = params.has_key("min_corr") ? d("min_corr") : p->library_min_corr;
    if (params.has_key("free_fit_box_min_corr")) {
      p->free_fit_box_min_corr = d("free_fit_box_min_corr");
    }
    if (params.has_key("free_fit_peak_height_fraction")) {
      p->free_fit_peak_height_fraction = d("free_fit_peak_height_fraction");
    }
    if (params.has_key("build_library")) {
      p->build_library = b("build_library");
    }
    read_line_shape(params, *p);
    return p;
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

  // Returns the profile fitter, or None for summation alone, so that a profile
  // library it built outlives the call (see tof_deconvolve_overlaps)
  std::shared_ptr<ProfileFitter> integrate_reflection_table_wrapper(
    dials::af::reflection_table& reflection_table,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    const bool& apply_lorentz,
    int n_threads,
    object profile_1d_params_obj,
    object profile_3d_params_obj,
    double lorentz_wavelength_power,
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

    std::shared_ptr<ProfileFitter> profile_fitter = make_profile_fitter(
      profile_1d_params_obj, profile_3d_params_obj, profile_3d_lp_params_obj);

    integrate_reflection_table(reflection_table,
                               experiment,
                               data,
                               incident_params,
                               absorption_params,
                               apply_lorentz,
                               n_threads,
                               profile_fitter,
                               lorentz_wavelength_power);
    return profile_fitter;
  }

  std::size_t tof_deconvolve_overlaps_wrapper(
    dials::af::reflection_table& reflection_table,
    dxtbx::model::Experiment& experiment,
    dxtbx::ImageSequence& data,
    object incident_params_obj,
    object absorption_params_obj,
    const bool& apply_lorentz,
    double lorentz_wavelength_power,
    int n_threads,
    std::shared_ptr<ProfileFitter> profile_fitter,
    double intensity_scale,
    bool replace_all) {
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
    return tof_deconvolve_overlaps(reflection_table,
                                   experiment,
                                   data,
                                   incident_params,
                                   absorption_params,
                                   apply_lorentz,
                                   lorentz_wavelength_power,
                                   n_threads,
                                   profile_fitter,
                                   intensity_scale,
                                   replace_all);
  }

  /*
   * One weighted solve for the intensities of a cluster, from a dense matrix
   * of profiles (voxels by reflections). Returns the values and their
   * variances; every variance is -1 if the data do not determine them.
   */
  boost::python::tuple fit_overlap_cluster_wrapper(
    scitbx::af::const_ref<double, scitbx::af::c_grid<2>> profiles,
    scitbx::af::const_ref<double> data,
    scitbx::af::const_ref<double> variance) {
    const std::size_t n = profiles.accessor()[0];
    const std::size_t K = profiles.accessor()[1];
    DIALS_ASSERT(data.size() == n && variance.size() == n);

    OverlapClusterDesign design;
    design.n_components = static_cast<int>(K);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = 0; k < K; ++k) {
        if (profiles(i, k) != 0.0) {
          design.add(static_cast<int>(k), profiles(i, k));
        }
      }
      design.end_row();
    }
    OverlapClusterFit fit =
      solve_overlap_cluster(design,
                            std::vector<double>(data.begin(), data.end()),
                            std::vector<double>(variance.begin(), variance.end()));

    scitbx::af::shared<double> value(K, 0.0), value_variance(K, -1.0);
    if (fit.success) {
      for (std::size_t k = 0; k < K; ++k) {
        value[k] = fit.value[k];
        value_variance[k] = fit.covariance(k, k);
      }
    }
    return boost::python::make_tuple(value, value_variance);
  }

  // The coupled pairs as two parallel arrays of reflection indices
  boost::python::tuple tof_mask_neighbour_foreground_wrapper(
    dials::af::reflection_table& reflection_table) {
    std::vector<std::pair<std::size_t, std::size_t>> coupled =
      tof_mask_neighbour_foreground(reflection_table);
    scitbx::af::shared<std::size_t> first(coupled.size());
    scitbx::af::shared<std::size_t> second(coupled.size());
    for (std::size_t i = 0; i < coupled.size(); ++i) {
      first[i] = coupled[i].first;
      second[i] = coupled[i].second;
    }
    return boost::python::make_tuple(first, second);
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
    TOFProfile1DParams& profile_params_1d) {
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
                                                 profile_params_1d);
  }

  boost::python::tuple calculate_line_profile_for_reflection_3d_line_wrapper(
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
    TOFProfile3DParams& profile_params_3d) {
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
                                                    profile_params_3d);
  }

  // A line shape at times tof, for tests: kind "ibix", "ic" or "gutmann",
  // its parameters (LineShape::p), and the reference time of the peak
  scitbx::af::shared<double> line_shape_values(std::string kind,
                                               boost::python::list params,
                                               scitbx::af::shared<double> tof,
                                               double t_ref) {
    LineShape line;
    line.kind = kind == "ic"        ? LineShape::IC
                : kind == "gutmann" ? LineShape::GUTMANN
                                    : LineShape::IBIX;
    DIALS_ASSERT(boost::python::len(params) == line.n_params());
    for (int j = 0; j < line.n_params(); ++j) {
      line.p[j] = extract<double>(params[j]);
    }
    return line.evaluate(tof.const_ref(), t_ref);
  }

  /*
   * The 3D fits with an iBIX line shape on plain arrays, for tests. Voxels indexed
   * (ix * ny + iy) * nz + iz; a shape is (alpha, beta, sigma, sig_x, sig_y,
   * sig_p, drift_x, drift_y) and a position (x, y, t).
   */
  Profile3DData make_profile_3d_data(int nx,
                                     int ny,
                                     int x0,
                                     int y0,
                                     scitbx::af::shared<double> tof,
                                     scitbx::af::shared<double> value,
                                     scitbx::af::shared<double> gain,
                                     scitbx::af::shared<double> background,
                                     scitbx::af::shared<double> extra_variance,
                                     double background_level_variance) {
    Profile3DData d;
    d.nx = nx;
    d.ny = ny;
    d.nz = tof.size();
    for (int ix = 0; ix < nx; ++ix)
      d.x.push_back(x0 + ix + 0.5);
    for (int iy = 0; iy < ny; ++iy)
      d.y.push_back(y0 + iy + 0.5);
    d.t.assign(tof.begin(), tof.end());
    d.value.assign(value.begin(), value.end());
    d.gain.assign(gain.begin(), gain.end());
    d.background.assign(background.begin(), background.end());
    d.extra_variance.assign(extra_variance.begin(), extra_variance.end());
    d.background_level_variance = background_level_variance;
    DIALS_ASSERT(d.consistent());
    return d;
  }

  Profile3DShape shape_from_tuple(tuple s) {
    DIALS_ASSERT(len(s) == 8);
    Profile3DShape shape;
    shape.line = LineShape::ibix(
      extract<double>(s[0]), extract<double>(s[1]), extract<double>(s[2]));
    shape.sig_x = extract<double>(s[3]);
    shape.sig_y = extract<double>(s[4]);
    shape.sig_p = extract<double>(s[5]);
    shape.drift_x = extract<double>(s[6]);
    shape.drift_y = extract<double>(s[7]);
    shape.valid = true;
    return shape;
  }

  Profile3DPosition position_from_tuple(tuple p) {
    DIALS_ASSERT(len(p) == 3);
    Profile3DPosition position;
    position.x = extract<double>(p[0]);
    position.y = extract<double>(p[1]);
    position.t = extract<double>(p[2]);
    return position;
  }

  dict fit_result_as_dict(const Profile3DFitResult& r) {
    dict out;
    out["success"] = r.success;
    out["intensity"] = r.intensity;
    out["variance"] = r.variance;
    out["correlation"] = r.correlation;
    out["chi2"] = r.chi2;
    out["shape"] = make_tuple(r.shape.line.p[0],
                              r.shape.line.p[1],
                              r.shape.line.p[2],
                              r.shape.sig_x,
                              r.shape.sig_y,
                              r.shape.sig_p,
                              r.shape.drift_x,
                              r.shape.drift_y);
    out["position"] = make_tuple(r.position.x, r.position.y, r.position.t);
    return out;
  }

  scitbx::af::shared<double> profile_3d_ibix_profile_wrapper(
    int nx,
    int ny,
    int x0,
    int y0,
    scitbx::af::shared<double> tof,
    tuple shape,
    tuple position) {
    std::size_t n = nx * ny * tof.size();
    scitbx::af::shared<double> zeros(n, 0.0);
    Profile3DData d =
      make_profile_3d_data(nx, ny, x0, y0, tof, zeros, zeros, zeros, zeros, 0.0);
    std::vector<double> out;
    bool ok = profile_3d_detail::model(
      d, shape_from_tuple(shape), position_from_tuple(position), out);
    DIALS_ASSERT(ok);
    scitbx::af::shared<double> result(out.size(), 0.0);
    std::copy(out.begin(), out.end(), result.begin());
    return result;
  }

  dict fit_profile_3d_ibix_wrapper(int nx,
                                   int ny,
                                   int x0,
                                   int y0,
                                   scitbx::af::shared<double> tof,
                                   scitbx::af::shared<double> value,
                                   scitbx::af::shared<double> gain,
                                   scitbx::af::shared<double> background,
                                   scitbx::af::shared<double> extra_variance,
                                   double background_level_variance,
                                   const TOFProfile3DParams& params,
                                   dict options) {
    // options: shape (without it, every parameter is fitted), seed,
    // tolerance, fix_position, scatter (broadens the shape)
    object shape = options.get("shape");
    object seed = options.get("seed");
    object tolerance = options.get("tolerance");
    object scatter = options.get("scatter");
    bool fix_position = extract<bool>(options.get("fix_position", false));
    Profile3DData d = make_profile_3d_data(nx,
                                           ny,
                                           x0,
                                           y0,
                                           tof,
                                           value,
                                           gain,
                                           background,
                                           extra_variance,
                                           background_level_variance);
    if (shape.is_none()) {
      return fit_result_as_dict(
        fit_profile_3d_shape(d, params, params.library_min_corr));
    }
    Profile3DPosition seed_position, tolerance_position;
    const Profile3DPosition* seed_ptr = nullptr;
    const Profile3DPosition* tolerance_ptr = nullptr;
    if (!seed.is_none()) {
      seed_position = position_from_tuple(extract<tuple>(seed));
      seed_ptr = &seed_position;
    }
    if (!tolerance.is_none()) {
      tolerance_position = position_from_tuple(extract<tuple>(tolerance));
      tolerance_ptr = &tolerance_position;
    }
    Profile3DShape fixed_shape = shape_from_tuple(extract<tuple>(shape));
    if (!scatter.is_none()) {
      fixed_shape = broaden_profile_3d_shape(
        fixed_shape, position_from_tuple(extract<tuple>(scatter)));
    }
    if (fix_position) {
      DIALS_ASSERT(seed_ptr != nullptr);
      return fit_result_as_dict(fit_profile_3d_fixed(d, fixed_shape, *seed_ptr));
    }
    return fit_result_as_dict(
      fit_profile_3d_forced(d, fixed_shape, seed_ptr, tolerance_ptr));
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
    class_<TOFProfile1DParams>("TOFProfile1DParams", no_init)
      .def("__init__", make_constructor(&make_TOFProfile1DParams))
      .def_readwrite("library_min_i_sigma", &TOFProfile1DParams::library_min_i_sigma)
      .def_readwrite("library_min_corr", &TOFProfile1DParams::library_min_corr);

    class_<TOFProfile3DParams, std::shared_ptr<TOFProfile3DParams>>(
      "TOFProfile3DParams", no_init)
      .def("__init__", make_constructor(&make_TOFProfile3DParams))
      .def_readwrite("library_min_i_sigma", &TOFProfile3DParams::library_min_i_sigma)
      .def_readwrite("library_min_corr", &TOFProfile3DParams::library_min_corr)
      .def_readwrite("position_tolerance", &TOFProfile3DParams::position_tolerance);

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

    def("profile_3d_ibix_profile",
        &profile_3d_ibix_profile_wrapper,
        (arg("nx"),
         arg("ny"),
         arg("x0"),
         arg("y0"),
         arg("tof"),
         arg("shape"),
         arg("position")));

    def("line_shape_values",
        &line_shape_values,
        (arg("kind"), arg("params"), arg("tof"), arg("t_ref")));

    def("fit_profile_3d_ibix",
        &fit_profile_3d_ibix_wrapper,
        (arg("nx"),
         arg("ny"),
         arg("x0"),
         arg("y0"),
         arg("tof"),
         arg("value"),
         arg("gain"),
         arg("background"),
         arg("extra_variance"),
         arg("background_level_variance"),
         arg("params"),
         arg("options") = dict()));

    class_<ProfileFitter, std::shared_ptr<ProfileFitter>, boost::noncopyable>(
      "TOFProfileFitter", no_init)
      .def("library_size", &ProfileFitter::library_size)
      .def("library_size_xy", &ProfileFitter::library_size_xy);

    def("tof_deconvolve_overlaps",
        &tof_deconvolve_overlaps_wrapper,
        (arg("reflection_table"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("apply_lorentz_correction"),
         arg("lorentz_wavelength_power"),
         arg("n_threads"),
         arg("profile_fitter"),
         arg("intensity_scale") = 1.0,
         arg("replace_all") = false));

    def("fit_overlap_cluster",
        &fit_overlap_cluster_wrapper,
        (arg("profiles"), arg("data"), arg("variance")));

    def("tof_assign_overlap_ownership",
        &tof_assign_overlap_ownership,
        (arg("reflection_table")));

    def("tof_mask_neighbour_foreground",
        &tof_mask_neighbour_foreground_wrapper,
        (arg("reflection_table")));

    def("integrate_reflection_table",
        &integrate_reflection_table_wrapper,
        (arg("reflection_table"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("apply_lorentz_correction"),
         arg("n_threads"),
         arg("profile_1d_params") = object(),
         arg("profile_3d_params") = object(),
         arg("lorentz_wavelength_power") = 4.0,
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
         arg("profile_params_1d")));

    def("calculate_line_profile_for_reflection_3d",
        &calculate_line_profile_for_reflection_3d_line_wrapper,
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
         arg("profile_params_3d")));

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
