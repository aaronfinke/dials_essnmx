#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_LP_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_LP_H

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <boost/optional.hpp>
#include "tof_profile_3d_bvg.h"
#include "tof_utils.h"
#include <scitbx/constants.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <dials/array_family/scitbx_shared_and_versa.h>

/*
3D profile fitting for a long-pulse spallation source (e.g. ESS NMX), using a
separable long-pulse (ToF) x bivariate Gaussian (BVG) model (XY).

At a long-pulse source the proton pulse (2.86 ms at ESS) dominates the width
of a peak in ToF, so the peak is a flat-topped trapezoid rather than the
single-peaked moderator pulse the other profile methods describe. The ToF
line shape is

  rectangle (width W, the proton pulse)
    (x) exponential (time constant tau, moderator emission)
    (x) Gaussian (sigma, everything else: mosaicity, divergence, timing)

which has the closed form

  g(t) = [F(t - T0) - F(t - T0 - W)] / W

where F is the CDF of an exponentially modified Gaussian. g has unit area, so
W -> 0 reduces to the EMG density, i.e. the short-pulse case. The fitted
amplitude is solved analytically by weighted least squares, as for the other
separable 3D methods.

Unlike the other 3D methods the ToF axis here is in microseconds (from the
shoebox's ToF bin centres), not in frames, so that W, tau and sigma are
physical, transferable instrument constants. Each bin's model value is the
mean of g over the bin (Gauss-Legendre), so coarse or unevenly-spaced bins
are handled.

tau and W are degenerate with T0 and sigma on the leading edge of the
trapezoid, so by default both are held fixed at calibrated values
(optimize_emission, optimize_pulse_width).
*/

namespace dials { namespace algorithms {

  /*
   * Holds params required for TOFProfile3DLP. All times are in microseconds.
   */
  struct TOFProfile3DLPParams {
    // Long-pulse ToF parameters
    double pulse_width;         // proton pulse length W (plateau width)
    bool optimize_pulse_width;  // If false W is held at pulse_width
    double pulse_width_min_factor, pulse_width_max_factor;  // W bounds / W0
    double tau, tau_min, tau_max;        // moderator emission time constant
    bool optimize_emission;              // If false tau is held at tau
    double sigma, sigma_min, sigma_max;  // Gaussian broadening

    // BVG parameters
    double SigX_min, SigX_max;
    double SigY_min, SigY_max;
    double SigP, SigP_min, SigP_max;  // spatial correlation

    // Control
    int n_restarts;              // number of attempts when fitting
    bool optimize_profile;       // If false the profile is generated with input params
    double max_drift_factor;     // Scales the per-reflection dXdt/dYdt bounds
    bool show_profile_failures;  // Prints debugging information

    // Acceptance of a free fit
    double min_correlation;          // of the ToF and XY projections with the data
    double plateau_height_fraction;  // allowed relative plateau height mismatch

    // Library fits (see fit_profile_3d_lp_library)
    bool library_mode;                 // fit with per-reflection library shapes
    double free_position_min_i_sigma;  // I/sigma at which the position is freed
    double position_tolerance;         // freed position range, x offset scatter

    TOFProfile3DLPParams(double pulse_width_,
                         bool optimize_pulse_width_,
                         double pulse_width_min_factor_,
                         double pulse_width_max_factor_,
                         double tau_,
                         double tau_min_,
                         double tau_max_,
                         bool optimize_emission_,
                         double sigma_,
                         double sigma_min_,
                         double sigma_max_,
                         double SigX_min_,
                         double SigX_max_,
                         double SigY_min_,
                         double SigY_max_,
                         double SigP_,
                         double SigP_min_,
                         double SigP_max_,
                         int n_restarts_,
                         bool optimize_profile_,
                         double max_drift_factor_,
                         bool show_profile_failures_,
                         double min_correlation_ = 0.75,
                         double plateau_height_fraction_ = 0.25,
                         bool library_mode_ = false,
                         double free_position_min_i_sigma_ = 10.0,
                         double position_tolerance_ = 3.0)
        : pulse_width(pulse_width_),
          optimize_pulse_width(optimize_pulse_width_),
          pulse_width_min_factor(pulse_width_min_factor_),
          pulse_width_max_factor(pulse_width_max_factor_),
          tau(tau_),
          tau_min(tau_min_),
          tau_max(tau_max_),
          optimize_emission(optimize_emission_),
          sigma(sigma_),
          sigma_min(sigma_min_),
          sigma_max(sigma_max_),
          SigX_min(SigX_min_),
          SigX_max(SigX_max_),
          SigY_min(SigY_min_),
          SigY_max(SigY_max_),
          SigP(SigP_),
          SigP_min(SigP_min_),
          SigP_max(SigP_max_),
          n_restarts(n_restarts_),
          optimize_profile(optimize_profile_),
          max_drift_factor(max_drift_factor_),
          show_profile_failures(show_profile_failures_),
          min_correlation(min_correlation_),
          plateau_height_fraction(plateau_height_fraction_),
          library_mode(library_mode_),
          free_position_min_i_sigma(free_position_min_i_sigma_),
          position_tolerance(position_tolerance_) {}
  };

  /*
   * What a fit measured, for the library and for diagnostics. The centre is
   * absolute: detector pixels, and the ToF (us) of the centroid of the
   * long-pulse shape, T0 + W/2 + tau.
   */
  struct LPFitResult {
    double W = 0, tau = 0, sigma = 0;
    double SigX = 0, SigY = 0, SigP = 0, dXdt = 0, dYdt = 0;
    double centre_x = 0, centre_y = 0, centre_t = 0;
    double cc_tof = 0, cc_xy = 0;
    bool position_free = true;  // false where a library fit held it fixed
  };

  /*
   * A library shape for one reflection, and where it is expected: its
   * prediction corrected by the library's median offset on the panel, with
   * the scatter of the library's own offsets about those medians.
   */
  struct LPLibraryShape {
    double W = 0, tau = 0, sigma = 0;
    double SigX = 0, SigY = 0, SigP = 0, dXdt = 0, dYdt = 0;
    double expected_x = 0, expected_y = 0, expected_t = 0;  // px, px, us
    double scatter_x = 0, scatter_y = 0, scatter_t = 0;     // px, px, us
  };

  // Smallest values allowed for the long-pulse widths (us); all three appear
  // in denominators
  const double LP_MIN_WIDTH = 1e-6;
  const double LP_MIN_PULSE_WIDTH = 1e-3;

  // exp(z^2) erfc(z) for z >= 0, using the asymptotic expansion where
  // exp(z^2) would overflow
  inline double lp_erfcx_nonneg(double z) {
    if (z > 25.0) {
      const double iz2 = 1.0 / (z * z);
      return (1.0 - 0.5 * iz2 + 0.75 * iz2 * iz2)
             / (z * std::sqrt(scitbx::constants::pi));
    }
    return std::exp(z * z) * std::erfc(z);
  }

  /*
   * CDF of an exponentially modified Gaussian (Gaussian sigma convolved with
   * an exponential of time constant tau):
   *
   *   F(t) = Phi(u) - 1/2 erfc(z) exp(-t/tau + v^2)
   *   u = t / (sqrt(2) sigma), v = sigma / (sqrt(2) tau), z = v - u
   *
   * The second term is evaluated on one of two branches by the sign of z:
   * for z >= 0 exp(-t/tau + v^2) can overflow while erfc(z) underflows, so
   * the scaled erfcx(z) is used, and the remaining exponent simplifies
   * exactly to -u^2 (since 2uv = t/tau); for z < 0 erfcx itself overflows,
   * but erfc(z) lies in (1, 2) and the exponent v^2 - t/tau = v(v - 2u) is
   * negative.
   */
  inline double lp_emg_cdf(double t, double tau, double sigma) {
    tau = std::max(tau, LP_MIN_WIDTH);
    sigma = std::max(sigma, LP_MIN_WIDTH);
    const double u = t / (std::sqrt(2.0) * sigma);
    const double v = sigma / (std::sqrt(2.0) * tau);
    const double z = v - u;
    const double phi = 0.5 * std::erfc(-u);
    double tail;
    if (z >= 0.0) {
      tail = 0.5 * lp_erfcx_nonneg(z) * std::exp(-u * u);
    } else {
      tail = 0.5 * std::erfc(z) * exp_safe(v * v - t / tau);
    }
    const double F = phi - tail;
    if (!std::isfinite(F)) return (t > 0.0) ? 1.0 : 0.0;
    return std::min(std::max(F, 0.0), 1.0);
  }

  // Unit-area long-pulse line shape at time t (us)
  inline double lp_func(double t, double T0, double W, double tau, double sigma) {
    W = std::max(W, LP_MIN_PULSE_WIDTH);
    const double val =
      (lp_emg_cdf(t - T0, tau, sigma) - lp_emg_cdf(t - T0 - W, tau, sigma)) / W;
    return (std::isfinite(val) && val > 0.0) ? val : 0.0;
  }

  // Bin edges (us) from bin centres: midpoints between centres, with the
  // outer edges extrapolated by half the neighbouring spacing
  inline scitbx::af::shared<double> lp_bin_edges(scitbx::af::const_ref<double> tof) {
    const std::size_t n = tof.size();
    scitbx::af::shared<double> edges(n + 1, 0.0);
    if (n == 0) return edges;
    if (n == 1) {
      edges[0] = tof[0] - 0.5;
      edges[1] = tof[0] + 0.5;
      return edges;
    }
    for (std::size_t i = 1; i < n; ++i)
      edges[i] = 0.5 * (tof[i - 1] + tof[i]);
    edges[0] = tof[0] - 0.5 * (tof[1] - tof[0]);
    edges[n] = tof[n - 1] + 0.5 * (tof[n - 1] - tof[n - 2]);
    return edges;
  }

  /*
   * Long-pulse line shape for bins centred at tof (us): the mean of the
   * unit-area shape over each bin, by 5-point Gauss-Legendre quadrature
   */
  inline scitbx::af::shared<double> lp_raw(scitbx::af::const_ref<double> tof,
                                           double T0,
                                           double W,
                                           double tau,
                                           double sigma) {
    static const double gl_x[5] = {-0.9061798459386640,
                                   -0.5384693101056831,
                                   0.0,
                                   0.5384693101056831,
                                   0.9061798459386640};
    static const double gl_w[5] = {0.2369268850561891,
                                   0.4786286704993665,
                                   0.5688888888888889,
                                   0.4786286704993665,
                                   0.2369268850561891};

    const std::size_t n = tof.size();
    scitbx::af::shared<double> out(n, 0.0);
    const scitbx::af::shared<double> edges = lp_bin_edges(tof);
    for (std::size_t i = 0; i < n; ++i) {
      const double a = edges[i];
      const double b = edges[i + 1];
      const double half = 0.5 * (b - a);
      const double mid = 0.5 * (a + b);
      double acc = 0.0;
      for (int k = 0; k < 5; ++k)
        acc += gl_w[k] * lp_func(mid + half * gl_x[k], T0, W, tau, sigma);
      const double val = 0.5 * acc;  // mean over the bin: (1/2) sum w_k g(x_k)
      out[i] = (std::isfinite(val) && val > 0.0) ? val : 0.0;
    }
    return out;
  }

  // Spatial (BVG) part of the long-pulse model
  struct LPSpatial {
    double SigX = 1.0, SigY = 1.0, SigP = 0.0;  // widths (px) and correlation
    double dXdt = 0.0, dYdt = 0.0;              // centre drift per us
    double X0 = 0.0, Y0 = 0.0;                  // centre (px) rel. to the peak voxel
  };

  struct Profile3DLPFunctor {
    const scitbx::af::versa<vec3<double>, af::c_grid<3>>
      coords;  // (x, y) centred at peak; z unused (frames)
    const scitbx::af::versa<double, af::c_grid<3>> intensities;
    const scitbx::af::versa<double, af::c_grid<3>> background_variances;
    // Corrected intensity per raw count, per voxel. The intensities are
    // corrected (Lorentz, spectrum), so a voxel with model value m holds
    // m / gain counts and its variance is gain * m + var_background.
    const scitbx::af::versa<double, af::c_grid<3>> gains;
    scitbx::af::shared<double> tof_axis;  // ToF (us) centred at peak, per z
    std::array<double, 11> min_bounds, max_bounds;
    int num_data_points, num_params;
    // Per-voxel weights, held fixed during one Levenberg-Marquardt run and
    // updated from the model between runs (iteratively reweighted fitting)
    scitbx::af::shared<double> weights;

    /*
     * Parameter mapping:
     *   x[0] = T0           pulse onset (us), relative to the centred peak
     *   x[1] = log(W)       plateau (proton pulse) width (us)
     *   x[2] = log(tau)     moderator emission time constant (us)
     *   x[3] = log(sigma)   Gaussian broadening (us)
     *   x[4] = log(SigX)    BVG sigma x
     *   x[5] = log(SigY)    BVG sigma y
     *   x[6] = SigP         BVG correlation (-1,1), clamped
     *   x[7] = dXdt         BVG centre drift in x per us (peak walk)
     *   x[8] = dYdt         BVG centre drift in y per us (peak walk)
     *   x[9] = X0           BVG centre x (px), relative to the peak voxel
     *   x[10] = Y0          BVG centre y (px), relative to the peak voxel
     * A parameter held fixed has equal min and max bounds, so clamp_params
     * pins it and df gives it a zero Jacobian column.
     */

    Profile3DLPFunctor(
      const scitbx::af::versa<vec3<double>, af::c_grid<3>> coords_,
      const scitbx::af::versa<double, af::c_grid<3>> intensities_,
      const scitbx::af::versa<double, af::c_grid<3>> background_variances_,
      const scitbx::af::versa<double, af::c_grid<3>> gains_,
      const scitbx::af::shared<double> tof_axis_,
      const std::array<double, 11>& minb,
      const std::array<double, 11>& maxb)
        : coords(coords_),
          intensities(intensities_),
          background_variances(background_variances_),
          gains(gains_),
          tof_axis(tof_axis_) {
      min_bounds = minb;
      max_bounds = maxb;
      num_data_points = coords.size();
      num_params = 11;
      DIALS_ASSERT(tof_axis.size() == coords.accessor()[2]);
      set_data_weights();
    }

    int values() const {
      return num_data_points;
    }
    int inputs() const {
      return num_params;
    }

    Eigen::VectorXd clamp_params(const Eigen::VectorXd& x) const {
      Eigen::VectorXd xc = x;
      for (int i = 0; i < num_params; ++i)
        xc[i] = std::min(std::max(x[i], min_bounds[i]), max_bounds[i]);
      return xc;
    }

    void extract_params(const Eigen::VectorXd& xc,
                        double& T0,
                        double& W,
                        double& tau,
                        double& sigma,
                        LPSpatial& sp) const {
      T0 = xc[0];
      W = std::exp(xc[1]);
      tau = std::exp(xc[2]);
      sigma = std::exp(xc[3]);
      sp.SigX = std::exp(xc[4]);
      sp.SigY = std::exp(xc[5]);
      sp.SigP = xc[6];
      sp.dXdt = xc[7];
      sp.dYdt = xc[8];
      sp.X0 = xc[9];
      sp.Y0 = xc[10];
    }

    static double clean_var(double var_b) {
      return (std::isfinite(var_b) && var_b > 0.0) ? var_b : 0.0;
    }

    /*
     * Weights. Weights from the observed counts (Neyman) bias a weighted
     * least-squares amplitude low by about one count per voxel; minimising a
     * chi-squared whose weights follow the model (Pearson) biases the shape
     * towards a broader profile and the intensity high. Both matter here,
     * because a long-pulse peak spreads its counts over many ToF bins and
     * counts per voxel are low. So the weights come from the model but are
     * held fixed during each minimisation and updated between them, which
     * converges to the quasi-likelihood (Poisson score) solution.
     */
    double gain(int i) const {
      const double g = gains[i];
      return (std::isfinite(g) && g > 0.0) ? g : 1.0;
    }

    // Weight for a voxel whose expected signal (corrected units) is m: one
    // over gain * m + var_background, floored at the variance of one count
    double weight_for(int i, double m) const {
      const double g = gain(i);
      return 1.0
             / std::max(g * std::max(m, 0.0) + clean_var(background_variances[i]),
                        g * g);
    }

    void set_data_weights() {
      weights.resize(num_data_points);
      for (int i = 0; i < num_data_points; ++i)
        weights[i] = weight_for(i, std::abs(intensities[i]));
    }

    void set_model_weights(const scitbx::af::shared<double>& model) {
      weights.resize(num_data_points);
      for (int i = 0; i < num_data_points; ++i)
        weights[i] = weight_for(i, model[i]);
    }

    // Unscaled model value (ToF shape x BVG) at voxel (ix, iy, iz)
    double unit_model(const scitbx::af::shared<double>& tof_shape,
                      int ix,
                      int iy,
                      int iz,
                      const LPSpatial& sp) const {
      const double dt = tof_axis[iz];
      const double dx = coords(ix, iy, iz)[0] - sp.X0 - sp.dXdt * dt;
      const double dy = coords(ix, iy, iz)[1] - sp.Y0 - sp.dYdt * dt;
      return tof_shape[iz] * bvg_func(dx, dy, sp.SigX, sp.SigY, sp.SigP);
    }

    // Unscaled model over the whole box, in c_grid order
    scitbx::af::shared<double> unit_model_box(
      const scitbx::af::shared<double>& tof_shape,
      const LPSpatial& sp) const {
      const int nx = coords.accessor()[0];
      const int ny = coords.accessor()[1];
      const int nz = coords.accessor()[2];
      scitbx::af::shared<double> p(num_data_points, 0.0);
      int count = 0;
      for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
          for (int iz = 0; iz < nz; ++iz)
            p[count++] = unit_model(tof_shape, ix, iy, iz, sp);
      return p;
    }

    scitbx::af::shared<double> unit_model_box(const Eigen::VectorXd& x) const {
      const Eigen::VectorXd xc = clamp_params(x);
      double T0, W, tau, sigma;
      LPSpatial sp;
      extract_params(xc, T0, W, tau, sigma, sp);
      const scitbx::af::shared<double> tof_shape =
        lp_raw(tof_axis.const_ref(), T0, W, tau, sigma);
      return unit_model_box(tof_shape, sp);
    }

    /*
     * Analytical amplitude via weighted least squares with the current
     * weights. Optionally returns sum(p) and sum(w p^2), the sums needed for
     * the variance of the intensity.
     */
    double calc_scale(const scitbx::af::shared<double>& p,
                      double* sum_p = nullptr,
                      double* sum_wpp = nullptr) const {
      double num = 0.0, den = 0.0, sp = 0.0;
      for (int i = 0; i < num_data_points; ++i) {
        num += weights[i] * intensities[i] * p[i];
        den += weights[i] * p[i] * p[i];
        sp += p[i];
      }
      if (sum_p) *sum_p = sp;
      if (sum_wpp) *sum_wpp = den;
      const double S = (den > 0.0) ? (num / den) : 0.0;
      return std::isfinite(S) ? S : 0.0;
    }

    // Model (Scale * p) for params x with the current weights
    scitbx::af::shared<double> model_box(const Eigen::VectorXd& x) const {
      scitbx::af::shared<double> p = unit_model_box(x);
      const double Scale = calc_scale(p);
      for (int i = 0; i < num_data_points; ++i)
        p[i] *= Scale;
      return p;
    }

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
      const scitbx::af::shared<double> model = model_box(x);
      fvec.resize(num_data_points);
      for (int i = 0; i < num_data_points; ++i) {
        const double diff = (intensities[i] - model[i]) * std::sqrt(weights[i]);
        fvec[i] = std::isfinite(diff) ? diff : 1e6;
      }
      return 0;
    }

    // Finite-difference Jacobian (forward differences)
    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
      J.resize(num_data_points, num_params);
      const double eps = 1e-5;
      const Eigen::VectorXd xc = clamp_params(x);
      Eigen::VectorXd f0(num_data_points);
      operator()(xc, f0);
      for (int j = 0; j < num_params; ++j) {
        Eigen::VectorXd xp = xc;
        const double delta = eps * (1.0 + std::abs(xc[j]));
        xp[j] += delta;
        const Eigen::VectorXd xpc = clamp_params(xp);
        Eigen::VectorXd fp(num_data_points);
        operator()(xpc, fp);
        const double step = xpc[j] - xc[j];
        if (std::abs(step) < 1e-8)
          J.col(j).setZero();
        else
          J.col(j) = (fp - f0) / step;
      }
      return 0;
    }
  };

  class TOFProfile3DLP {
  public:
    const scitbx::af::versa<vec3<double>, af::c_grid<3>> coords;
    const scitbx::af::versa<double, af::c_grid<3>> intensities;
    const scitbx::af::versa<double, af::c_grid<3>> background_variances;
    const scitbx::af::versa<double, af::c_grid<3>> gains;
    scitbx::af::versa<double, af::c_grid<3>> y_norm;
    scitbx::af::shared<double> tof_axis;  // ToF (us) relative to the peak
    int n_restarts;
    Eigen::VectorXd params;
    std::array<double, 11> min_bounds, max_bounds;
    double tof_range;
    boost::optional<Profile3DLPFunctor> functor;
    // Absolute position the relative coordinates are measured from: the
    // brightest voxel (px) and the ToF of the brightest slice (us)
    double ref_x = 0.0, ref_y = 0.0, ref_t = 0.0;
    double min_correlation, plateau_height_fraction;
    // Projection correlations of the last accepted or checked fit
    mutable double last_cc_tof = 0.0, last_cc_xy = 0.0;

    TOFProfile3DLP(scitbx::af::const_ref<vec3<double>, af::c_grid<3>> coords_,
                   const scitbx::af::versa<double, af::c_grid<3>> intensities_,
                   const scitbx::af::versa<double, af::c_grid<3>> background_variances_,
                   const scitbx::af::versa<double, af::c_grid<3>> gains_,
                   scitbx::af::const_ref<double> tof_z,
                   const TOFProfile3DLPParams& p)
        : coords(get_rel_coords_3d(coords_, intensities_)),
          intensities(intensities_),
          background_variances(background_variances_),
          gains(gains_),
          n_restarts(p.n_restarts),
          min_correlation(p.min_correlation),
          plateau_height_fraction(p.plateau_height_fraction) {
      {
        // Same rule as get_rel_coords_3d: the brightest finite voxel
        double best = -1.0;
        std::size_t idx = 0;
        for (std::size_t i = 0; i < intensities_.size(); ++i) {
          const double v = intensities_[i];
          if (std::isfinite(v) && v > best) {
            best = v;
            idx = i;
          }
        }
        ref_x = coords_[idx][0];
        ref_y = coords_[idx][1];
      }
      const int nx = coords.accessor()[0];
      const int ny = coords.accessor()[1];
      const int nz = coords.accessor()[2];
      DIALS_ASSERT(tof_z.size() == nz);
      DIALS_ASSERT(nz > 0);

      const int n = intensities.size();
      y_norm.resize(intensities.accessor());
      for (int i = 0; i < n; ++i) {
        const double v = intensities[i];
        y_norm[i] = (std::isfinite(v) && v > 0.0) ? v : 0.0;
      }

      // ToF projection, used to centre the ToF axis and seed T0 and W
      scitbx::af::shared<double> proj(nz, 0.0);
      for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
          for (int iz = 0; iz < nz; ++iz)
            proj[iz] += y_norm(ix, iy, iz);
      const std::size_t iz_peak =
        std::distance(proj.begin(), std::max_element(proj.begin(), proj.end()));
      tof_axis.resize(nz);
      for (int iz = 0; iz < nz; ++iz)
        tof_axis[iz] = tof_z[iz] - tof_z[iz_peak];
      ref_t = tof_z[iz_peak];
      const double tof_min = tof_axis[0];
      const double tof_max = tof_axis[nz - 1];
      tof_range = std::abs(tof_max - tof_min);

      // Half-maximum crossings of the projection: the leading edge seeds T0
      // and, when W is refined, the crossing separation seeds W
      double tL = tof_min, tR = tof_max;
      half_max_crossings(tof_axis.const_ref(), proj.const_ref(), iz_peak, tL, tR);

      // Pulse width (plateau) bounds; held fixed unless optimize_pulse_width
      const double W0 = std::max(p.pulse_width, LP_MIN_PULSE_WIDTH);
      double W_lo = W0, W_hi = W0, W_init = W0;
      if (p.optimize_pulse_width) {
        W_lo = std::max(p.pulse_width_min_factor * W0, LP_MIN_PULSE_WIDTH);
        W_hi = std::max(p.pulse_width_max_factor * W0, W_lo);
        W_init = std::min(std::max(tR - tL, W_lo), W_hi);
        if (!(tR - tL > 0.0)) W_init = std::min(std::max(W0, W_lo), W_hi);
      }

      // Emission time constant; held fixed unless optimize_emission
      const double tau_lo_b = std::max(p.tau_min, LP_MIN_WIDTH);
      const double tau_hi_b = std::max(p.tau_max, tau_lo_b);
      const double tau0 = std::min(std::max(p.tau, tau_lo_b), tau_hi_b);
      const double tau_lo = p.optimize_emission ? tau_lo_b : tau0;
      const double tau_hi = p.optimize_emission ? tau_hi_b : tau0;

      const double sigma_lo = std::max(p.sigma_min, LP_MIN_WIDTH);
      const double sigma_hi = std::max(p.sigma_max, sigma_lo);
      const double sigma_init = std::min(std::max(p.sigma, sigma_lo), sigma_hi);

      // T0: the half-maximum of the leading edge sits close to T0 + tau, so
      // seed T0 there; T0 may lie before the box (a peak clipped by the box)
      const double T0_min = tof_min - W_hi;
      const double T0_max = tof_max;
      const double T0_init = std::min(std::max(tL - tau0, T0_min), T0_max);

      const std::array<double, 2> SigX_bounds = {p.SigX_min, p.SigX_max};
      const std::array<double, 2> SigY_bounds = {p.SigY_min, p.SigY_max};
      const std::array<double, 2> SigP_bounds = {p.SigP_min, p.SigP_max};
      const BVGSpatialInit sp = estimate_bvg_spatial_init(coords,
                                                          y_norm,
                                                          p.SigP,
                                                          SigX_bounds,
                                                          SigY_bounds,
                                                          SigP_bounds,
                                                          tof_range,
                                                          p.max_drift_factor);

      /*
       * Spatial start: centre, widths and correlation from the moments of the
       * XY projection over the pixels holding at least 10% of its maximum.
       * The BVG centre is free (X0, Y0) because the brightest voxel, which
       * the coordinates are measured from, can sit well off the centre of an
       * extended or streaked spot.
       */
      double X0_init = 0.0, Y0_init = 0.0;
      double SigX_init = sp.SigX_init, SigY_init = sp.SigY_init;
      double SigP_init = sp.SigP_init;
      double x_lo = 0.0, x_hi = 0.0, y_lo = 0.0, y_hi = 0.0;
      {
        scitbx::af::shared<double> proj_xy(nx * ny, 0.0);
        for (int ix = 0; ix < nx; ++ix)
          for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz) {
              const double v = intensities(ix, iy, iz);
              if (std::isfinite(v)) proj_xy[ix * ny + iy] += v;
            }
        const double pmax = *std::max_element(proj_xy.begin(), proj_xy.end());
        double w = 0.0, wx = 0.0, wy = 0.0, wxx = 0.0, wyy = 0.0, wxy = 0.0;
        x_lo = x_hi = coords(0, 0, 0)[0];
        y_lo = y_hi = coords(0, 0, 0)[1];
        for (int ix = 0; ix < nx; ++ix)
          for (int iy = 0; iy < ny; ++iy) {
            const double x = coords(ix, iy, 0)[0];
            const double y = coords(ix, iy, 0)[1];
            x_lo = std::min(x_lo, x);
            x_hi = std::max(x_hi, x);
            y_lo = std::min(y_lo, y);
            y_hi = std::max(y_hi, y);
            const double v = proj_xy[ix * ny + iy];
            if (!(pmax > 0.0) || v < 0.1 * pmax) continue;
            w += v;
            wx += v * x;
            wy += v * y;
            wxx += v * x * x;
            wyy += v * y * y;
            wxy += v * x * y;
          }
        if (w > 0.0) {
          X0_init = wx / w;
          Y0_init = wy / w;
          const double vx = wxx / w - X0_init * X0_init;
          const double vy = wyy / w - Y0_init * Y0_init;
          const double cxy = wxy / w - X0_init * Y0_init;
          if (vx > 0.0 && vy > 0.0) {
            SigX_init =
              std::min(std::max(std::sqrt(vx), SigX_bounds[0]), SigX_bounds[1]);
            SigY_init =
              std::min(std::max(std::sqrt(vy), SigY_bounds[0]), SigY_bounds[1]);
            SigP_init = std::min(std::max(cxy / std::sqrt(vx * vy), SigP_bounds[0]),
                                 SigP_bounds[1]);
          }
        }
      }

      min_bounds = {T0_min,
                    std::log(W_lo),
                    std::log(tau_lo),
                    std::log(sigma_lo),
                    std::log(std::max(SigX_bounds[0], LP_MIN_WIDTH)),
                    std::log(std::max(SigY_bounds[0], LP_MIN_WIDTH)),
                    SigP_bounds[0],
                    -sp.dXdt_max,
                    -sp.dYdt_max,
                    x_lo,
                    y_lo};
      max_bounds = {T0_max,
                    std::log(W_hi),
                    std::log(tau_hi),
                    std::log(sigma_hi),
                    std::log(std::max(SigX_bounds[1], LP_MIN_WIDTH)),
                    std::log(std::max(SigY_bounds[1], LP_MIN_WIDTH)),
                    SigP_bounds[1],
                    sp.dXdt_max,
                    sp.dYdt_max,
                    x_hi,
                    y_hi};

      params.resize(11);
      params[0] = T0_init;
      params[1] = std::log(W_init);
      params[2] = std::log(tau0);
      params[3] = std::log(sigma_init);
      params[4] = std::log(SigX_init);
      params[5] = std::log(SigY_init);
      params[6] = SigP_init;
      params[7] = 0.0;  // dXdt: no drift assumed unless the fit finds evidence
      params[8] = 0.0;  // dYdt
      params[9] = X0_init;
      params[10] = Y0_init;

      // The fit uses the signed (background-subtracted) intensities: clipping
      // negative voxels to zero, as y_norm does for the initial estimates,
      // would bias the amplitude high at low counts
      scitbx::af::versa<double, af::c_grid<3>> y_fit(intensities.accessor());
      for (int i = 0; i < n; ++i) {
        const double v = intensities[i];
        y_fit[i] = std::isfinite(v) ? v : 0.0;
      }
      DIALS_ASSERT(gains.accessor().all_eq(intensities.accessor()));
      functor.emplace(
        coords, y_fit, background_variances, gains, tof_axis, min_bounds, max_bounds);
      params = functor->clamp_params(params);
    }

    static void half_max_crossings(scitbx::af::const_ref<double> t,
                                   scitbx::af::const_ref<double> y,
                                   std::size_t imax,
                                   double& tL,
                                   double& tR) {
      tL = t[0];
      tR = t[t.size() - 1];
      if (t.size() < 2) return;
      const double half_max = 0.5 * y[imax];
      if (!(half_max > 0.0)) return;
      for (std::size_t i = imax; i-- > 0;) {
        if (y[i] <= half_max && y[i + 1] > half_max) {
          const double frac = (half_max - y[i]) / (y[i + 1] - y[i]);
          tL = t[i] + frac * (t[i + 1] - t[i]);
          break;
        }
      }
      for (std::size_t i = imax; i + 1 < y.size(); ++i) {
        if (y[i] > half_max && y[i + 1] <= half_max) {
          const double frac = (half_max - y[i]) / (y[i + 1] - y[i]);
          tR = t[i] + frac * (t[i + 1] - t[i]);
          break;
        }
      }
    }

    // Param getters (times in us)
    double get_T0() const {
      return params[0];
    }
    double get_W() const {
      return std::exp(params[1]);
    }
    double get_tau() const {
      return std::exp(params[2]);
    }
    double get_sigma() const {
      return std::exp(params[3]);
    }
    double get_SigX() const {
      return std::exp(params[4]);
    }
    double get_SigY() const {
      return std::exp(params[5]);
    }
    double get_SigP() const {
      return params[6];
    }
    double get_dXdt() const {
      return params[7];
    }
    double get_dYdt() const {
      return params[8];
    }
    double get_X0() const {
      return params[9];
    }
    double get_Y0() const {
      return params[10];
    }

    scitbx::af::shared<double> tof_shape() const {
      return lp_raw(tof_axis.const_ref(), get_T0(), get_W(), get_tau(), get_sigma());
    }

    // Unscaled model over the box for the current params
    scitbx::af::shared<double> unit_model_box() const {
      return functor->unit_model_box(params);
    }

    // Fitted amplitude and the sums needed for the intensity variance, for
    // the current params
    double scale(double* sum_p = nullptr, double* sum_wpp = nullptr) const {
      return functor->calc_scale(unit_model_box(), sum_p, sum_wpp);
    }

    // Reweights from the model at the current params
    void update_weights() {
      functor->set_model_weights(functor->model_box(params));
    }

    scitbx::af::versa<double, af::c_grid<3>> result() const {
      const scitbx::af::shared<double> p = unit_model_box();
      const double Scale = functor->calc_scale(p);
      scitbx::af::versa<double, af::c_grid<3>> out(coords.accessor());
      for (std::size_t i = 0; i < out.size(); ++i) {
        const double val = Scale * p[i];
        out[i] = std::isfinite(val) ? val : 0.0;
      }
      return out;
    }

    // Profile-fitted intensity: the model summed over the box, Scale * sum(p)
    double calc_intensity() const {
      double sum_p = 0.0;
      const double Scale = scale(&sum_p);
      const double I = Scale * sum_p;
      return std::isfinite(I) ? I : 0.0;
    }

    /*
     * Variance of the profile-fitted intensity, propagated from the
     * weighted least-squares amplitude: with I = Scale * sum(p) and
     * var(Scale) = 1 / sum(w p^2), var(I) = sum(p)^2 / sum(w p^2). The
     * uncertainty of the shape parameters is not included.
     */
    double calc_variance() const {
      double sum_p = 0.0, sum_wpp = 0.0;
      scale(&sum_p, &sum_wpp);
      if (!(sum_wpp > 0.0)) return 0.0;
      const double v = sum_p * sum_p / sum_wpp;
      return std::isfinite(v) ? v : 0.0;
    }

    // Reduced chi-squared of the current fit
    double calc_reduced_chi2() const {
      Eigen::VectorXd fvec(functor->num_data_points);
      (*functor)(params, fvec);
      int n_free = 1;  // the amplitude
      for (int j = 0; j < functor->num_params; ++j)
        if (max_bounds[j] > min_bounds[j]) ++n_free;
      const int dof = std::max(functor->num_data_points - n_free, 1);
      return fvec.squaredNorm() / dof;
    }

    /*
     * One fit from x_init: a minimisation with weights from the data, then
     * n_reweight more, each with the weights recomputed from the previous
     * model. On success params holds the solution and the weights follow it.
     */
    bool minimise(const Eigen::VectorXd& x_init,
                  double& final_error,
                  int maxfev = 500,
                  double xtol = 1e-8,
                  double ftol = 1e-8,
                  bool start_from_data_weights = true) {
      typedef Eigen::LevenbergMarquardt<Profile3DLPFunctor, double> LM;
      const int n_reweight = 3;
      Eigen::VectorXd x = functor->clamp_params(x_init);
      if (start_from_data_weights) {
        functor->set_data_weights();
      } else {
        functor->set_model_weights(functor->model_box(x));
      }
      for (int it = 0; it <= n_reweight; ++it) {
        if (it > 0) functor->set_model_weights(functor->model_box(x));
        LM lm(*functor);
        lm.parameters.maxfev = maxfev;
        lm.parameters.xtol = xtol;
        lm.parameters.ftol = ftol;
        const int result = lm.minimize(x);
        if (result < 0) return false;
        x = functor->clamp_params(x);
      }
      functor->set_model_weights(functor->model_box(x));
      Eigen::VectorXd fvec(functor->num_data_points);
      (*functor)(x, fvec);
      final_error = fvec.squaredNorm();
      params = x;
      return true;
    }

    void set_bounds(const std::array<double, 11>& minb,
                    const std::array<double, 11>& maxb) {
      min_bounds = minb;
      max_bounds = maxb;
      functor->min_bounds = minb;
      functor->max_bounds = maxb;
    }

    // Absolute centre of the fitted peak: px, px and the ToF (us) of the
    // centroid of the long-pulse shape
    void centre(double& x, double& y, double& t) const {
      x = ref_x + get_X0();
      y = ref_y + get_Y0();
      t = ref_t + get_T0() + 0.5 * get_W() + get_tau();
    }

    LPFitResult fit_result() const {
      LPFitResult r;
      r.W = get_W();
      r.tau = get_tau();
      r.sigma = get_sigma();
      r.SigX = get_SigX();
      r.SigY = get_SigY();
      r.SigP = get_SigP();
      r.dXdt = get_dXdt();
      r.dYdt = get_dYdt();
      centre(r.centre_x, r.centre_y, r.centre_t);
      r.cc_tof = last_cc_tof;
      r.cc_xy = last_cc_xy;
      return r;
    }

    /*
     * Fit with a library shape. First at the expected position with only the
     * amplitude free, the shape broadened in x, y and t by the scatter of the
     * library's own peaks about their expected positions, since this peak's
     * position is known no better than that. A weak peak's counts cannot
     * place it: a position fitted to them finds the brightest noise and reads
     * the intensity high. Where that first fit reaches
     * free_position_min_i_sigma, the position (T0, X0, Y0) is fitted as well,
     * with the shape unbroadened, within position_tolerance times the scatter
     * (at least 1 px and one ToF bin) of the expected position. Every finite
     * fit is accepted: the intensity of a weak peak may be negative.
     */
    bool fit_library(const LPLibraryShape& lib,
                     double free_position_min_i_sigma,
                     double position_tolerance,
                     bool& position_free) {
      position_free = false;
      const double W = std::max(lib.W, LP_MIN_PULSE_WIDTH);
      const double tau = std::max(lib.tau, LP_MIN_WIDTH);
      const double sigma = std::max(lib.sigma, LP_MIN_WIDTH);
      const double SigX = std::max(lib.SigX, LP_MIN_WIDTH);
      const double SigY = std::max(lib.SigY, LP_MIN_WIDTH);
      const double SigP = std::min(std::max(lib.SigP, -0.99), 0.99);
      const double X0 = lib.expected_x - ref_x;
      const double Y0 = lib.expected_y - ref_y;
      const double T0 = lib.expected_t - ref_t - 0.5 * W - tau;

      auto pinned = [&](double sx, double sy, double st) {
        std::array<double, 11> v = {T0,
                                    std::log(W),
                                    std::log(tau),
                                    std::log(std::sqrt(sigma * sigma + st * st)),
                                    std::log(std::sqrt(SigX * SigX + sx * sx)),
                                    std::log(std::sqrt(SigY * SigY + sy * sy)),
                                    SigP,
                                    lib.dXdt,
                                    lib.dYdt,
                                    X0,
                                    Y0};
        return v;
      };

      // 1. Amplitude only, at the expected position, broadened shape
      const std::array<double, 11> broad =
        pinned(lib.scatter_x, lib.scatter_y, lib.scatter_t);
      set_bounds(broad, broad);
      for (int i = 0; i < 11; ++i)
        params[i] = broad[i];
      functor->set_data_weights();
      for (int it = 0; it < 5; ++it)
        update_weights();
      const double I1 = calc_intensity();
      const double V1 = calc_variance();
      if (!std::isfinite(I1) || !(V1 > 0.0)) return false;
      if (!(I1 / std::sqrt(V1) >= free_position_min_i_sigma)) return true;

      // 2. Position free near the expected one, unbroadened shape
      const std::array<double, 11> sharp = pinned(0.0, 0.0, 0.0);
      double bin = 0.0;
      if (tof_axis.size() > 1)
        bin =
          std::abs(tof_axis[tof_axis.size() - 1] - tof_axis[0]) / (tof_axis.size() - 1);
      const double dx = std::max(position_tolerance * lib.scatter_x, 1.0);
      const double dy = std::max(position_tolerance * lib.scatter_y, 1.0);
      const double dt = std::max(position_tolerance * lib.scatter_t, bin);
      std::array<double, 11> lo = sharp, hi = sharp;
      lo[0] = T0 - dt;
      hi[0] = T0 + dt;
      lo[9] = X0 - dx;
      hi[9] = X0 + dx;
      lo[10] = Y0 - dy;
      hi[10] = Y0 + dy;
      set_bounds(lo, hi);
      Eigen::VectorXd x0(11);
      for (int i = 0; i < 11; ++i)
        x0[i] = sharp[i];
      double err = 0.0;
      const Eigen::VectorXd keep = params;
      if (minimise(x0, err, 200, 1e-8, 1e-8, false)) {
        const double I2 = calc_intensity();
        const double V2 = calc_variance();
        if (std::isfinite(I2) && V2 > 0.0) {
          position_free = true;
          return true;
        }
      }
      // The freed fit failed: keep the fixed-position one
      set_bounds(broad, broad);
      params = keep;
      for (int it = 0; it < 5; ++it)
        update_weights();
      return true;
    }

    bool fit(bool show_profile_failures,
             int maxfev = 500,
             double xtol = 1e-8,
             double ftol = 1e-8) {
      auto run_single_fit = [&](const Eigen::VectorXd& x_init,
                                double& final_error) -> bool {
        return minimise(x_init, final_error, maxfev, xtol, ftol);
      };

      const Eigen::VectorXd x0 = params;
      double fit_resid = std::numeric_limits<double>::infinity();
      bool success = run_single_fit(x0, fit_resid);

      if (success) {
        const double I_prf = this->calc_intensity();
        if (this->trust_result(fit_resid, I_prf, show_profile_failures)) return true;
      }

      std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
      std::normal_distribution<double> norm_dist(0.0, 0.5);

      // Log-scale params: W(1), tau(2), sigma(3), SigX(4), SigY(5). Fixed ones
      // are pinned again by clamp_params.
      const int n_log = 5;
      const int log_idx[5] = {1, 2, 3, 4, 5};

      // T0, SigP, dXdt/dYdt and X0/Y0 are perturbed additively, scaled by their own
      // bound width
      auto perturb_position = [&](Eigen::VectorXd& x_try, double scale) {
        for (int j : {0, 6, 7, 8, 9, 10}) {
          x_try[j] += norm_dist(rng) * scale * (max_bounds[j] - min_bounds[j]);
        }
      };

      for (int i = 0; i < n_restarts; ++i) {
        Eigen::VectorXd x_try = x0;

        if (i < n_restarts / 3) {
          for (int j = 0; j < n_log; ++j)
            x_try[log_idx[j]] += std::log(0.8 + 0.4 * unit_dist(rng));
          perturb_position(x_try, 0.05);
        } else if (i < 2 * n_restarts / 3) {
          const double scale = 0.5 + 1.5 * unit_dist(rng);
          for (int j = 0; j < n_log; ++j)
            x_try[log_idx[j]] += std::log(scale);
          perturb_position(x_try, 0.15);
        } else {
          for (int j = 0; j < functor->num_params; ++j)
            x_try[j] = min_bounds[j] + unit_dist(rng) * (max_bounds[j] - min_bounds[j]);
        }
        x_try = functor->clamp_params(x_try);

        success = run_single_fit(x_try, fit_resid);
        if (!success) continue;
        const double I_prf = this->calc_intensity();
        if (this->trust_result(fit_resid, I_prf, show_profile_failures)) return true;
      }
      return false;
    }

    bool trust_result(double error, double I_prf, bool show_error = false) const {
      if (!std::isfinite(error) || error <= 0.0) {
        if (show_error)
          std::cerr << "profile_3d_lp fitting failure: invalid error (" << error
                    << ")\n";
        return false;
      }
      if (!(I_prf > 0.0)) {
        if (show_error)
          std::cerr << "profile_3d_lp fitting failure: intensity too small (" << I_prf
                    << ")\n";
        return false;
      }
      const double SigP = get_SigP();
      if (std::abs(SigP) >= 1.0) {
        if (show_error)
          std::cerr << "profile_3d_lp fitting failure: |SigP|>=1 (" << SigP << ")\n";
        return false;
      }

      const scitbx::af::versa<double, af::c_grid<3>> r = result();
      const double limit = 1e12;
      for (std::size_t i = 0; i < r.size(); ++i) {
        if (!std::isfinite(r[i]) || r[i] > limit) {
          if (show_error)
            std::cerr << "profile_3d_lp fitting failure: model value non-finite or "
                         "exceeds limit\n";
          return false;
        }
      }

      /*
       * The shape checks are made on the ToF projection (summed over x, y)
       * and the XY projection (summed over ToF), not voxel by voxel: a
       * long-pulse peak spreads its counts over many ToF bins, so single
       * voxels are dominated by counting noise even for a well-measured
       * reflection.
       */
      const int nx = coords.accessor()[0];
      const int ny = coords.accessor()[1];
      const int nz = coords.accessor()[2];
      scitbx::af::shared<double> d_t(nz, 0.0), m_t(nz, 0.0);
      scitbx::af::shared<double> d_xy(nx * ny, 0.0), m_xy(nx * ny, 0.0);
      for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
          for (int iz = 0; iz < nz; ++iz) {
            // signed data, so the projections are background-subtracted sums
            const double d = intensities(ix, iy, iz);
            const double m = r(ix, iy, iz);
            if (!std::isfinite(d)) continue;
            d_t[iz] += d;
            m_t[iz] += m;
            d_xy[ix * ny + iy] += d;
            m_xy[ix * ny + iy] += m;
          }

      // In ToF the contrast is between the plateau, where the model is flat,
      // and the background either side, so the whole projection is used. In
      // XY the correlation is taken where the model is significant, so that
      // background-only pixels (pure noise) do not make a weak but
      // well-fitted reflection look like a failed fit.
      const double corr_t = correlation(d_t.const_ref(), m_t.const_ref(), 0.0);
      const double corr_xy = correlation(d_xy.const_ref(), m_xy.const_ref(), 0.05);
      last_cc_tof = corr_t;
      last_cc_xy = corr_xy;
      if (corr_t < min_correlation || corr_xy < min_correlation) {
        if (show_error)
          std::cerr << "profile_3d_lp fitting failure: low correlation (ToF " << corr_t
                    << ", XY " << corr_xy << ")\n";
        return false;
      }

      // Plateau height of the ToF projection: compare the model's maximum
      // with the data's mean over the bins where the model is within 10% of
      // its maximum, so one noisy bin cannot fail the check
      const double m_peak = *std::max_element(m_t.begin(), m_t.end());
      double d_plateau = 0.0;
      int n_plateau = 0;
      for (int iz = 0; iz < nz; ++iz) {
        if (m_t[iz] >= 0.9 * m_peak) {
          d_plateau += d_t[iz];
          ++n_plateau;
        }
      }
      if (n_plateau > 0) d_plateau /= n_plateau;
      if (plateau_height_fraction > 0.0 && d_plateau > 0.0
          && std::abs(m_peak - d_plateau) > plateau_height_fraction * d_plateau) {
        if (show_error)
          std::cerr << "profile_3d_lp fitting failure: plateau height mismatch "
                       "(profile="
                    << m_peak << ", data=" << d_plateau << ")\n";
        return false;
      }
      return true;
    }

    // Pearson correlation of data a and model b over the elements where the
    // model is at least min_frac of its maximum
    static double correlation(scitbx::af::const_ref<double> a,
                              scitbx::af::const_ref<double> b,
                              double min_frac) {
      const std::size_t n = a.size();
      if (n < 2) return 0.0;
      const double b_max = *std::max_element(b.begin(), b.end());
      if (!(b_max > 0.0)) return 0.0;
      const double cut = min_frac * b_max;
      double ma = 0.0, mb = 0.0;
      std::size_t m = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (b[i] < cut) continue;
        ma += a[i];
        mb += b[i];
        ++m;
      }
      if (m < 3) return 0.0;
      ma /= m;
      mb /= m;
      double num = 0.0, da = 0.0, db = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        if (b[i] < cut) continue;
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma) * (a[i] - ma);
        db += (b[i] - mb) * (b[i] - mb);
      }
      return num / std::sqrt(da * db + 1e-12);
    }
  };

  /*
   * Fits the long-pulse model to one shoebox. coords are (x, y, z) voxel
   * centres as filled by integrate_shoebox; gains the corrected intensity per
   * raw count of each voxel; tof_z gives the ToF (us) of each z slice. On
   * success I_prf_out and var_prf_out hold the profile-fitted intensity and
   * its variance.
   */
  inline bool fit_profile_3d_lp(
    scitbx::af::const_ref<vec3<double>, af::c_grid<3>> coords,
    const scitbx::af::versa<double, af::c_grid<3>> intensities,
    const scitbx::af::versa<double, af::c_grid<3>> background_variances,
    const scitbx::af::versa<double, af::c_grid<3>> gains,
    scitbx::af::const_ref<double> tof_z,
    TOFProfile3DLPParams& profile_params,
    double& I_prf_out,
    double& var_prf_out,
    boost::optional<scitbx::af::versa<double, af::c_grid<3>>> profile_3d_out =
      boost::none,
    bool update_params = false,
    LPFitResult* result_out = nullptr) {
    if (coords.size() == 0 || tof_z.size() != coords.accessor()[2]) return false;

    TOFProfile3DLP profile(
      coords, intensities, background_variances, gains, tof_z, profile_params);

    bool profile_success = true;
    if (profile_params.optimize_profile) {
      profile_success = profile.fit(profile_params.show_profile_failures);
    } else {
      // Shape fixed at the input params: only the amplitude is fitted,
      // reweighting from the model as fit() does
      for (int it = 0; it < 4; ++it)
        profile.update_weights();
    }

    if (profile_success) {
      I_prf_out = profile.calc_intensity();
      var_prf_out = profile.calc_variance();
      if (result_out) *result_out = profile.fit_result();

      if (update_params) {
        profile_params.pulse_width = profile.get_W();
        profile_params.tau = profile.get_tau();
        profile_params.sigma = profile.get_sigma();
        profile_params.SigP = profile.get_SigP();
      }

      if (profile_3d_out) {
        const scitbx::af::versa<double, af::c_grid<3>> pred = profile.result();
        scitbx::af::versa<double, af::c_grid<3>>& out = *profile_3d_out;
        DIALS_ASSERT(pred.accessor().all_eq(out.accessor()));
        for (std::size_t i = 0; i < pred.size(); ++i)
          out[i] = pred[i];
      }
    }
    return profile_success;
  }

  /*
   * Fits one shoebox with a library shape (see TOFProfile3DLP::fit_library).
   * Returns false only if no finite intensity could be fitted.
   */
  inline bool fit_profile_3d_lp_library(
    scitbx::af::const_ref<vec3<double>, af::c_grid<3>> coords,
    const scitbx::af::versa<double, af::c_grid<3>> intensities,
    const scitbx::af::versa<double, af::c_grid<3>> background_variances,
    const scitbx::af::versa<double, af::c_grid<3>> gains,
    scitbx::af::const_ref<double> tof_z,
    const TOFProfile3DLPParams& profile_params,
    const LPLibraryShape& library_shape,
    double& I_prf_out,
    double& var_prf_out,
    boost::optional<scitbx::af::versa<double, af::c_grid<3>>> profile_3d_out =
      boost::none,
    LPFitResult* result_out = nullptr) {
    if (coords.size() == 0 || tof_z.size() != coords.accessor()[2]) return false;

    TOFProfile3DLP profile(
      coords, intensities, background_variances, gains, tof_z, profile_params);
    bool position_free = false;
    if (!profile.fit_library(library_shape,
                             profile_params.free_position_min_i_sigma,
                             profile_params.position_tolerance,
                             position_free)) {
      return false;
    }
    I_prf_out = profile.calc_intensity();
    var_prf_out = profile.calc_variance();
    if (result_out) {
      *result_out = profile.fit_result();
      // Report the library shape, not the broadened one used for the fit
      result_out->sigma = library_shape.sigma;
      result_out->SigX = library_shape.SigX;
      result_out->SigY = library_shape.SigY;
      result_out->position_free = position_free;
    }
    if (profile_3d_out) {
      const scitbx::af::versa<double, af::c_grid<3>> pred = profile.result();
      scitbx::af::versa<double, af::c_grid<3>>& out = *profile_3d_out;
      DIALS_ASSERT(pred.accessor().all_eq(out.accessor()));
      for (std::size_t i = 0; i < pred.size(); ++i)
        out[i] = pred[i];
    }
    return std::isfinite(I_prf_out) && var_prf_out > 0.0;
  }

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_LP_H
