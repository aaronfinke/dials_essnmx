#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_TOF_PROFILE_1D_IBIX_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_TOF_PROFILE_1D_IBIX_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <array>
#include <cassert>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <random>
#include "tof_utils.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>

/*
1D profile fitting method based on
Yano, N., Yamada, T., Hosoya, T. et al.
Application of profile fitting method to neutron time-of-flight protein
single crystal diffraction data collected at the iBIX. Sci Rep 6,
36628 (2016). https://doi.org/10.1038/srep36628
*/
namespace dials { namespace algorithms {

  /*
   * A peak shape, as fitted on one reflection and reusable on another.  Held
   * separately from TOFProfile1DIBIXParams so that a library can be built
   * from many reflections without mutating shared fitting parameters.
   */
  struct IBIXShape {
    double alpha = 0.0;
    double beta = 0.0;
    double sigma = 0.0;
    bool valid = false;
  };

  /*
   * Holds params required for profile1d
   */
  struct TOFProfile1DIBIXParams {
    double A;
    double A_min;
    double A_max;
    double alpha;
    double alpha_min;
    double alpha_max;
    double beta;
    double beta_min;
    double beta_max;
    int n_restarts;              // number of attempts when fitting
    bool optimize_profile;       // If false the profile is generated with input params
    bool show_profile_failures;  // Prints debugging information
    double fwhm_smoothing_window;  // Time over which to smooth before estimating sigma
    double trust_min_corr;         // Correlation with the data a fit must reach
    double trust_peak_tolerance;   // How far the fitted peak may sit from the data's

    TOFProfile1DIBIXParams(double A_min,
                           double A_max,
                           double alpha,
                           double alpha_min,
                           double alpha_max,
                           double beta,
                           double beta_min,
                           double beta_max,
                           int n_restarts,
                           bool optimize_profile,
                           bool show_profile_failures,
                           double fwhm_smoothing_window,
                           double trust_min_corr,
                           double trust_peak_tolerance)

        : A(A_min),
          A_min(A_min),
          A_max(A_max),
          alpha(alpha),
          alpha_min(alpha_min),
          alpha_max(alpha_max),
          beta(beta),
          beta_min(beta_min),
          beta_max(beta_max),
          n_restarts(n_restarts),
          optimize_profile(optimize_profile),
          show_profile_failures(show_profile_failures),
          fwhm_smoothing_window(fwhm_smoothing_window),
          trust_min_corr(trust_min_corr),
          trust_peak_tolerance(trust_peak_tolerance) {}
  };

  static scitbx::af::shared<double> profile1d_func(scitbx::af::const_ref<double> tof,
                                                   double A,
                                                   double alpha,
                                                   double beta,
                                                   double sigma,
                                                   double T_ph) {
    /*
     * func used to generate the actual profile
     * (Numbers) refer to equations in https://doi.org/10.1038/srep36628
     */

    const size_t m = tof.size();
    scitbx::af::shared<double> out(m, 0.0);

    double sigma2 = sigma * sigma;
    double sigma_sqrt = std::sqrt(2.0 * sigma2);
    double N = (alpha * beta) / (2.0 * (alpha + beta));  // (5)

    for (size_t i = 0; i < m; ++i) {
      double dT = tof[i] - T_ph;                             // (11)
      double u = alpha * 0.5 * (alpha * sigma2 + 2.0 * dT);  // (7)
      double v = beta * 0.5 * (beta * sigma2 - 2.0 * dT);    // (8)
      double y = (alpha * sigma2 + dT) / sigma_sqrt;         // (9)
      double z = (beta * sigma2 - dT) / sigma_sqrt;          // (10)

      // Stable evaluation with erfcx
      double term1 = std::exp(u - y * y) * erfcx_safe(y);
      double term2 = std::exp(v - z * z) * erfcx_safe(z);

      double val = A * N * (term1 + term2);  // (1)
      if (!std::isfinite(val)) val = 1e-12;
      out[i] = val;
    }
    return out;
  }

  /*
   * The amplitude enters the model linearly, so for a given shape and position
   * the best A is a one-line least squares rather than something to search for:
   * A* = sum(y g) / sum(g g), with g the same profile at unit amplitude.  Used
   * to seed the fit, because until A is roughly right the model is flat
   * compared with the data and the gradient with respect to the peak position
   * all but vanishes, which leaves the optimiser unable to find the peak.
   */
  static double analytic_amplitude(scitbx::af::const_ref<double> tof,
                                   scitbx::af::const_ref<double> y_norm,
                                   double alpha,
                                   double beta,
                                   double sigma,
                                   double T_ph) {
    scitbx::af::shared<double> g = profile1d_func(tof, 1.0, alpha, beta, sigma, T_ph);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
      if (!is_finite_double(g[i])) {
        continue;
      }
      num += y_norm[i] * g[i];
      den += g[i] * g[i];
    }
    if (!(den > 0.0) || !is_finite_double(num)) {
      return 1.0;
    }
    double a = num / den;
    return is_finite_double(a) && a > 0.0 ? a : 1.0;
  }

  struct IBIXProfileFunctor {
    scitbx::af::const_ref<double> tof;
    scitbx::af::const_ref<double> y_norm;  // Assumed normalized
    std::array<double, 5> min_bounds;      // parameter bounds
    std::array<double, 5> max_bounds;      // parameter bounds
    int num_data_points, num_params;

    IBIXProfileFunctor(scitbx::af::const_ref<double> tof_,
                       scitbx::af::const_ref<double> y_norm_,
                       const std::array<double, 5>& minb,
                       const std::array<double, 5>& maxb)
        : tof(tof_), y_norm(y_norm_) {
      min_bounds = minb;
      max_bounds = maxb;
      num_data_points = tof.size();
      num_params = 5;
    }

    int values() const {
      return num_data_points;
    }

    int inputs() const {
      return num_params;
    }

    inline Eigen::VectorXd clamp_params(const Eigen::VectorXd& x) const {
      Eigen::VectorXd xc = x;
      for (int i = 0; i < x.size(); ++i) {
        xc[i] = std::min(std::max(x[i], min_bounds[i]), max_bounds[i]);
      }
      return xc;
    }

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
      Eigen::VectorXd xc = clamp_params(x);
      double A = xc[0];
      double alpha = xc[1];
      double beta = xc[2];
      double sigma = xc[3];
      double T_ph = xc[4];

      scitbx::af::shared<double> model =
        profile1d_func(tof, A, alpha, beta, sigma, T_ph);
      assert(model.size() == num_data_points);
      for (int i = 0; i < num_data_points; ++i) {
        fvec[i] = y_norm[i] - model[i];
      }
      return 0;
    }

    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
      const double eps = 1e-5;
      Eigen::VectorXd xc = clamp_params(x);
      J.resize(num_data_points, num_params);

      for (int j = 0; j < num_params; ++j) {
        // Perturb param
        double delta = eps * std::max(1.0, std::abs(xc[j]));
        Eigen::VectorXd xp = xc, xm = xc;
        xp[j] += delta;
        xm[j] -= delta;

        Eigen::VectorXd xpc = clamp_params(xp);
        Eigen::VectorXd xmc = clamp_params(xm);

        double step = xpc[j] - xmc[j];
        if (std::abs(step) < 1e-14) {
          J.col(j).setZero();
          continue;
        }

        Eigen::VectorXd fp(num_data_points), fm(num_data_points);
        operator()(xpc, fp);
        operator()(xmc, fm);

        // Central difference
        J.col(j) = (fp - fm) / step;
      }

      return 0;
    }
  };

  /*
   * As IBIXProfileFunctor, but with the peak shape held fixed and only the
   * amplitude and the peak position free.  Used when the shape comes from a
   * profile library built on strong reflections, as in Mantid's
   * BVGFitTools.doBVGFit with forceParams: a weak reflection has too few
   * counts to determine its own shape, but two parameters against the same
   * number of points are well constrained.
   */
  struct IBIXForcedFunctor {
    scitbx::af::const_ref<double> tof;
    scitbx::af::const_ref<double> y_norm;  // Assumed normalized
    double alpha, beta, sigma;             // fixed shape
    std::array<double, 2> min_bounds;      // bounds on A and T_ph
    std::array<double, 2> max_bounds;
    int num_data_points, num_params;

    IBIXForcedFunctor(scitbx::af::const_ref<double> tof_,
                      scitbx::af::const_ref<double> y_norm_,
                      double alpha_,
                      double beta_,
                      double sigma_,
                      const std::array<double, 2>& minb,
                      const std::array<double, 2>& maxb)
        : tof(tof_),
          y_norm(y_norm_),
          alpha(alpha_),
          beta(beta_),
          sigma(sigma_),
          min_bounds(minb),
          max_bounds(maxb) {
      num_data_points = tof.size();
      num_params = 2;
    }

    int values() const {
      return num_data_points;
    }

    int inputs() const {
      return num_params;
    }

    inline Eigen::VectorXd clamp_params(const Eigen::VectorXd& x) const {
      Eigen::VectorXd xc = x;
      for (int i = 0; i < x.size(); ++i) {
        xc[i] = std::min(std::max(x[i], min_bounds[i]), max_bounds[i]);
      }
      return xc;
    }

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
      Eigen::VectorXd xc = clamp_params(x);
      scitbx::af::shared<double> model =
        profile1d_func(tof, xc[0], alpha, beta, sigma, xc[1]);
      for (int i = 0; i < num_data_points; ++i) {
        fvec[i] = y_norm[i] - model[i];
      }
      return 0;
    }

    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
      const double eps = 1e-5;
      Eigen::VectorXd xc = clamp_params(x);
      J.resize(num_data_points, num_params);

      for (int j = 0; j < num_params; ++j) {
        double delta = eps * std::max(1.0, std::abs(xc[j]));
        Eigen::VectorXd xp = xc, xm = xc;
        xp[j] += delta;
        xm[j] -= delta;

        Eigen::VectorXd xpc = clamp_params(xp);
        Eigen::VectorXd xmc = clamp_params(xm);

        double step = xpc[j] - xmc[j];
        if (std::abs(step) < 1e-14) {
          J.col(j).setZero();
          continue;
        }

        Eigen::VectorXd fp(num_data_points), fm(num_data_points);
        operator()(xpc, fp);
        operator()(xmc, fm);
        J.col(j) = (fp - fm) / step;
      }

      return 0;
    }
  };

  class TOFProfile1DIBIX {
  public:
    scitbx::af::const_ref<double> tof;
    scitbx::af::const_ref<double> intensities;  // raw intensities
    scitbx::af::shared<double> y_norm;          // normalized intensities
    double intensity_max;
    int n_restarts;

    // params
    double A, alpha, beta, sigma, T_ph;
    double fwhm_smoothing_window;
    double trust_min_corr;
    double trust_peak_tolerance;
    std::array<double, 5> min_bounds;
    std::array<double, 5> max_bounds;

    TOFProfile1DIBIX(scitbx::af::const_ref<double> tof_,
                     scitbx::af::const_ref<double> intensities_,
                     double A_,
                     double alpha_,
                     double beta_,
                     double T_ph_,
                     const std::array<double, 2> A_bounds,
                     const std::array<double, 2> alpha_bounds,
                     const std::array<double, 2> beta_bounds,
                     int n_restarts_,
                     double fwhm_smoothing_window_,
                     double trust_min_corr_,
                     double trust_peak_tolerance_)
        : tof(tof_),
          intensities(intensities_),
          A(A_),
          alpha(alpha_),
          beta(beta_),
          sigma(1.0),
          T_ph(T_ph_),
          fwhm_smoothing_window(fwhm_smoothing_window_),
          trust_min_corr(trust_min_corr_),
          trust_peak_tolerance(trust_peak_tolerance_),
          n_restarts(n_restarts_) {
      DIALS_ASSERT(tof.size() > 0);
      DIALS_ASSERT(tof.size() == intensities.size());

      // Get max intensity
      intensity_max = 1.0;
      if (!(intensities.size() == 0)) {
        intensity_max = *std::max_element(intensities.begin(), intensities.end());
        if (intensity_max <= 0.0) intensity_max = 1.0;
      }

      // Get normalized y vector
      const size_t n = intensities.size();
      y_norm.resize(n);
      for (size_t i = 0; i < n; ++i) {
        double v = intensities[i];
        if (!is_finite_double(v)) v = 0.0;
        if (v < 0) v = 0.0;
        y_norm[i] = v / intensity_max;
      }

      // Set initial sigma from estimating peak width
      sigma = estimate_sigma_from_fwhm(tof, y_norm.const_ref());

      // Seed the amplitude at its analytic optimum for that shape
      double A_seed =
        analytic_amplitude(tof, y_norm.const_ref(), alpha, beta, sigma, T_ph);
      A = std::min(std::max(A_seed, A_bounds[0]), A_bounds[1]);

      // Param bounds (A, alpha, beta, sigma, T_ph)
      /*
       * The upper bound on sigma is taken from the span of the box rather than
       * a fixed time, so that it does not depend on how finely the data are
       * sliced.  A fixed 100 us bound is narrower than a real peak on some
       * instruments, and on finely sliced data it is the estimate below, not
       * sigma * 4, that sets the bound.
       */
      double tof_span = tof.back() - tof.front();
      min_bounds = {
        A_bounds[0], alpha_bounds[0], beta_bounds[0], sigma / 4.0, tof.front()};

      max_bounds = {A_bounds[1],
                    alpha_bounds[1],
                    beta_bounds[1],
                    std::max(tof_span / 4.0, sigma * 4.0),
                    tof.back()};

      // Sanity check params
      DIALS_ASSERT(A >= min_bounds[0] && A <= max_bounds[0]);
      DIALS_ASSERT(alpha >= min_bounds[1] && alpha <= max_bounds[1]);
      DIALS_ASSERT(beta >= min_bounds[2] && beta <= max_bounds[2]);
      DIALS_ASSERT(sigma >= min_bounds[3] && sigma <= max_bounds[3]);
      DIALS_ASSERT(T_ph >= min_bounds[4] && T_ph <= max_bounds[4]);
    }

    scitbx::af::shared<double> result() const {
      /*
       * Generates (unnormalized) profile for all positions in tof
       */

      scitbx::af::shared<double> m = profile1d_func(tof, A, alpha, beta, sigma, T_ph);
      for (auto& v : m)
        v *= intensity_max;
      return m;
    }

    double estimate_sigma_from_fwhm(scitbx::af::const_ref<double> tof,
                                    scitbx::af::const_ref<double> y) {
      /*
       * Estimates sigma param using full width at half maximum of peak in y
       */

      // Not enough data
      if (tof.size() <= 3) {
        return 1.0;
      }

      double mean_dt = (tof.back() - tof.front()) / std::max<size_t>(tof.size() - 1, 1);

      /*
       * The half-maximum search runs on a boxcar-smoothed copy of the
       * projection.  The window is a fixed time rather than a fixed number of
       * bins, so that the width estimated here does not depend on how finely
       * the data are sliced.  Without it, narrow bins hold few enough counts
       * that the tallest channel is a noise spike its neighbours fall to half
       * of within one bin: the estimate collapses onto mean_dt and the fit is
       * seeded several times too narrow.
       */
      scitbx::af::shared<double> y_smooth(y.size());
      int half_window = static_cast<int>(0.5 * fwhm_smoothing_window / mean_dt);
      if (half_window < 1) {
        std::copy(y.begin(), y.end(), y_smooth.begin());
      } else {
        int n = static_cast<int>(y.size());
        for (int i = 0; i < n; ++i) {
          int lo = std::max(0, i - half_window);
          int hi = std::min(n - 1, i + half_window);
          double total = 0.0;
          for (int j = lo; j <= hi; ++j) {
            total += y[j];
          }
          y_smooth[i] = total / (hi - lo + 1);
        }
      }
      scitbx::af::const_ref<double> ys = y_smooth.const_ref();

      // locate peak
      size_t imax = std::distance(ys.begin(), std::max_element(ys.begin(), ys.end()));
      double ymax = ys[imax];

      // Negative peak
      if (ymax <= 0.0) {
        return 1.0;
      }

      double half_max = 0.5 * ymax;

      // Search left crossing
      double tL = tof.front();
      for (size_t i = imax; i-- > 0;) {
        if (ys[i] <= half_max && ys[i + 1] > half_max) {
          double t0 = tof[i], t1 = tof[i + 1];
          double y0 = ys[i], y1 = ys[i + 1];
          double frac = (half_max - y0) / (y1 - y0);
          tL = t0 + frac * (t1 - t0);
          break;
        }
      }

      // Search right crossing
      double tR = tof.back();
      for (size_t i = imax; i + 1 < ys.size(); ++i) {
        if (ys[i] > half_max && ys[i + 1] <= half_max) {
          double t0 = tof[i], t1 = tof[i + 1];
          double y0 = ys[i], y1 = ys[i + 1];
          double frac = (half_max - y0) / (y1 - y0);
          tR = t0 + frac * (t1 - t0);
          break;
        }
      }

      // Full width at half maximum
      double fwhm = std::max(tR - tL, 0.0);
      DIALS_ASSERT(fwhm > 0.0);

      // 2.354520045 = approx(sqrt(2ln2))
      double sigma0 = fwhm / 2.354820045;

      // Unphysical sigma (check large as at least one sample spacing)
      sigma0 = std::max(sigma0, mean_dt);
      return sigma0;
    }

    double calc_intensity() const {
      /**
       * Get overall intensity with Simpsons rule then divide by mean_dt to
       * approximate summation scale
       */

      scitbx::af::shared<double> r = result();
      double mean_dt = (tof[tof.size() - 1] - tof[0]) / (tof.size() - 1);
      return simpson_integrate(r.const_ref(), tof) / mean_dt;
    }

    std::size_t get_max_profile_index() {
      /*
       * Returns the index of the max of the profile
       */

      auto profile_result = this->result();
      auto max_profile_it =
        std::max_element(profile_result.begin(), profile_result.end());
      std::size_t max_profile_index =
        std::distance(profile_result.begin(), max_profile_it);
      return max_profile_index;
    }

    bool fit(std::size_t max_sum_index,  // Peak index of the projected intensity
             bool show_profile_failures,
             int maxfev = 200,
             double xtol = 1e-8,
             double ftol = 1e-8) {
      /*
       * Least-squares minimization
       * Updates A, alpha, beta, sigma, T_ph
       * If fitting fails, params are perturbed n_restarts to find a solution
       */

      // Check enough data for fitting
      const int ndata = static_cast<int>(tof.size());
      if (ndata < 5) return false;

      IBIXProfileFunctor functor(tof, y_norm.const_ref(), min_bounds, max_bounds);
      typedef Eigen::LevenbergMarquardt<IBIXProfileFunctor, double> LM;

      auto run_single_fit = [&](const Eigen::VectorXd& x_init,
                                double& final_error) -> bool {
        LM lm(functor);
        lm.parameters.maxfev = maxfev;
        lm.parameters.xtol = xtol;
        lm.parameters.ftol = ftol;

        Eigen::VectorXd x = x_init;
        int result = lm.minimize(x);
        if (result < 0) return false;

        x = functor.clamp_params(x);

        // Compute residual norm
        Eigen::VectorXd fvec(functor.num_data_points);
        functor(x, fvec);
        final_error = fvec.squaredNorm();

        // Update fitted parameters
        A = x[0];
        alpha = x[1];
        beta = x[2];
        sigma = x[3];
        T_ph = x[4];

        return true;
      };

      // First fit attempt
      Eigen::VectorXd x0(5);
      x0 << A, alpha, beta, sigma, T_ph;
      double fit_resid = std::numeric_limits<double>::infinity();
      bool success = run_single_fit(x0, fit_resid);
      std::size_t max_profile_index;
      double I_prf, I_var;

      if (success) {
        I_prf = this->calc_intensity();
        if (this->trust_result(fit_resid,
                               I_prf,
                               max_sum_index,
                               max_profile_index,
                               show_profile_failures)) {
          return true;
        }
      }

      // Initial fit failed, perturb initial params
      std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

      for (int i = 0; i < n_restarts; ++i) {
        Eigen::VectorXd x_try(5);
        for (int j = 0; j < 5; ++j) {
          double span = max_bounds[j] - min_bounds[j];
          double rand_frac = (unit_dist(rng) - 0.5) * 0.4;
          double perturbed = x0[j] + rand_frac * span;
          x_try[j] = std::max(min_bounds[j], std::min(perturbed, max_bounds[j]));
        }

        // Attempt fit
        success = run_single_fit(x_try, fit_resid);
        if (!success) continue;

        I_prf = this->calc_intensity();
        max_profile_index = this->get_max_profile_index();
        if (this->trust_result(fit_resid,
                               I_prf,
                               max_sum_index,
                               max_profile_index,
                               show_profile_failures)) {
          return true;
        }
      }

      return false;
    }

    bool trust_result(double error,
                      double I_prf,
                      std::size_t max_sum_index,
                      std::size_t max_profile_index,
                      bool show_error = false) {
      /*
       * Tests to check the fit is reasonable
       */
      if (!std::isfinite(error) || error <= 0.0) {
        if (show_error) {
          std::cerr << "profile1d fitting failure: invalid error value (error=" << error
                    << ")\n";
        }
        return false;
      }

      // Check reasonable intensity
      if (I_prf < 1e-7) {
        if (show_error) {
          std::cerr << "profile1d fitting failure: profile intensity too small (I_prf="
                    << I_prf << ")\n";
        }
        return false;
      }

      /*
       * Check peak position close to data peak.  Measured as a time, not as a
       * number of bins: a fixed bin count means a tolerance that shrinks with
       * the slice width, which on finely sliced data rejects fits whose peak is
       * well within the width of the peak itself.
       */
      double peak_delta = std::abs(tof[max_profile_index] - tof[max_sum_index]);
      if (peak_delta > trust_peak_tolerance) {
        if (show_error) {
          std::cerr << "profile1d fitting failure: peak position mismatch (delta="
                    << peak_delta << " > " << trust_peak_tolerance << ")\n";
        }
        return false;
      }

      // Check peak isn't very flat
      auto m = result();
      double max_val = *std::max_element(m.begin(), m.end());
      double mean_val = std::accumulate(m.begin(), m.end(), 0.0) / m.size();
      double contrast = (max_val - mean_val) / (max_val + 1e-12);
      if (contrast < 0.1) {
        if (show_error) {
          std::cerr << "profile1d fitting failure: insufficient peak contrast "
                    << "(contrast=" << contrast << ", max_val=" << max_val
                    << ", mean_val=" << mean_val << ")\n";
        }
        return false;
      }

      // Check correlation with data
      double profile_peak = 0.0, data_peak = 0.0;
      double num = 0.0, denom_y = 0.0, denom_m = 0.0;

      for (std::size_t i = 0; i < tof.size(); ++i) {
        double y = y_norm[i];
        double p = m[i] / intensity_max;

        if (i == 0 || y > data_peak) {
          data_peak = y;
        }
        if (i == 0 || p > profile_peak) {
          profile_peak = p;
        }

        num += y * p;
        denom_y += y * y;
        denom_m += p * p;
      }

      double corr = num / std::sqrt(denom_y * denom_m + 1e-12);
      if (corr < trust_min_corr) {
        if (show_error) {
          std::cerr << "profile1d fitting failure: low correlation (corr=" << corr
                    << ")\n";
        }
        return false;
      }

      // Check peak height is within 10% of data peak
      double peak_diff = std::abs(profile_peak - data_peak);
      if (peak_diff > data_peak * 0.1) {
        if (show_error) {
          std::cerr << "profile1d fitting failure: peak height mismatch "
                    << "(profile_peak=" << profile_peak << ", data_peak=" << data_peak
                    << ", diff=" << peak_diff << ")\n";
        }
        return false;
      }

      return true;
    }
  };

  bool fit_profile_1d_ibix(
    scitbx::af::const_ref<double> projected_intensity,
    scitbx::af::const_ref<double> tof_z,
    TOFProfile1DIBIXParams& profile_params,
    double& I_prf_out,
    boost::optional<scitbx::af::shared<double>> line_profile_out = boost::none,
    bool update_params = false,
    IBIXShape* shape_out = nullptr) {
    /**
     * Wrapper for fitting a given reflection
     * If line_profile_out is provided the profile is returned at every
     * position in tof_z
     */

    // Get T_ph (peak position)
    auto max_it =
      std::max_element(projected_intensity.begin(), projected_intensity.end());
    size_t max_index = std::distance(projected_intensity.begin(), max_it);
    double T_ph = tof_z[max_index];

    // Fit profile
    const std::array<double, 2> A_bounds = {profile_params.A_min, profile_params.A_max};
    const std::array<double, 2> alpha_bounds = {profile_params.alpha_min,
                                                profile_params.alpha_max};
    const std::array<double, 2> beta_bounds = {profile_params.beta_min,
                                               profile_params.beta_max};

    TOFProfile1DIBIX profile(tof_z,
                             projected_intensity,
                             profile_params.A,
                             profile_params.alpha,
                             profile_params.beta,
                             T_ph,
                             A_bounds,
                             alpha_bounds,
                             beta_bounds,
                             profile_params.n_restarts,
                             profile_params.fwhm_smoothing_window,
                             profile_params.trust_min_corr,
                             profile_params.trust_peak_tolerance);

    bool profile_success = true;
    if (profile_params.optimize_profile) {
      profile_success = profile.fit(max_index, profile_params.show_profile_failures);
    }

    if (profile_success) {
      if (update_params) {
        profile_params.alpha = profile.alpha;
        profile_params.beta = profile.beta;
        profile_params.A = profile.A;
      }
      if (shape_out != nullptr) {
        shape_out->alpha = profile.alpha;
        shape_out->beta = profile.beta;
        shape_out->sigma = profile.sigma;
        shape_out->valid = true;
      }
      double I_prf = profile.calc_intensity();
      auto profile_result = profile.result();
      DIALS_ASSERT(projected_intensity.size() == profile_result.size());

      I_prf_out = I_prf;

      if (!line_profile_out) {
        return profile_success;
      }

      scitbx::af::shared<double> line_profile = *line_profile_out;
      DIALS_ASSERT(line_profile.size() == profile_result.size());
      for (std::size_t i = 0; i < profile_result.size(); ++i) {
        line_profile[i] = profile_result[i];
      }
      return profile_success;
    }
    return false;
  }

  /*
   * Fit a reflection with its peak shape taken from elsewhere, leaving only
   * the amplitude and the peak position free.
   *
   * Deliberately not gated on the correlation between model and data, unlike
   * the free fit: a forced fit has fewer parameters and so follows the noise
   * less closely, which lowers that correlation while raising the accuracy of
   * the intensity.  Gating on it would reject the better answer.
   */
  bool fit_profile_1d_ibix_forced(
    scitbx::af::const_ref<double> projected_intensity,
    scitbx::af::const_ref<double> tof_z,
    const IBIXShape& shape,
    double A_min,
    double A_max,
    double& I_prf_out,
    boost::optional<scitbx::af::shared<double>> line_profile_out = boost::none) {
    const int ndata = static_cast<int>(tof_z.size());
    if (ndata < 5 || !shape.valid) {
      return false;
    }
    if (!(shape.sigma > 0.0) || !(shape.alpha > 0.0) || !(shape.beta > 0.0)) {
      return false;
    }

    // Normalize, as the free fitter does, so that A is on a comparable scale
    double intensity_max =
      *std::max_element(projected_intensity.begin(), projected_intensity.end());
    if (!(intensity_max > 0.0)) {
      return false;
    }
    scitbx::af::shared<double> y_norm(ndata);
    for (int i = 0; i < ndata; ++i) {
      double v = projected_intensity[i];
      if (!is_finite_double(v) || v < 0.0) v = 0.0;
      y_norm[i] = v / intensity_max;
    }

    // Seed the peak position at the tallest channel
    std::size_t max_index = std::distance(
      projected_intensity.begin(),
      std::max_element(projected_intensity.begin(), projected_intensity.end()));

    std::array<double, 2> min_bounds = {A_min, tof_z.front()};
    std::array<double, 2> max_bounds = {A_max, tof_z.back()};

    IBIXForcedFunctor functor(tof_z,
                              y_norm.const_ref(),
                              shape.alpha,
                              shape.beta,
                              shape.sigma,
                              min_bounds,
                              max_bounds);

    double A_seed = analytic_amplitude(tof_z,
                                       y_norm.const_ref(),
                                       shape.alpha,
                                       shape.beta,
                                       shape.sigma,
                                       tof_z[max_index]);
    Eigen::VectorXd x(2);
    x << std::min(std::max(A_seed, A_min), A_max), tof_z[max_index];

    Eigen::LevenbergMarquardt<IBIXForcedFunctor, double> lm(functor);
    lm.parameters.maxfev = 200;
    lm.parameters.xtol = 1e-8;
    lm.parameters.ftol = 1e-8;
    if (lm.minimize(x) < 0) {
      return false;
    }
    x = functor.clamp_params(x);

    scitbx::af::shared<double> model =
      profile1d_func(tof_z, x[0], shape.alpha, shape.beta, shape.sigma, x[1]);
    for (auto& v : model) {
      v *= intensity_max;
      if (!is_finite_double(v)) {
        return false;
      }
    }

    double mean_dt = (tof_z[ndata - 1] - tof_z[0]) / (ndata - 1);
    if (!(mean_dt > 0.0)) {
      return false;
    }
    double I_prf = simpson_integrate(model.const_ref(), tof_z) / mean_dt;
    if (!is_finite_double(I_prf) || I_prf < 1e-7) {
      return false;
    }

    I_prf_out = I_prf;
    if (line_profile_out) {
      scitbx::af::shared<double> line_profile = *line_profile_out;
      if (line_profile.size() != model.size()) {
        return false;
      }
      for (std::size_t i = 0; i < model.size(); ++i) {
        line_profile[i] = model[i];
      }
    }
    return true;
  }

}}  // namespace dials::algorithms
#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_TOF_PROFILE_1D_IBIX_H */
