#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_PROFILELIB_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_PROFILELIB_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <dials/algorithms/integration/tof/tof_profile_1d_profilelib.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_bvg.h>

/*
profile_3d_profilelib fits profile_3d_ibix's model, the iBIX line shape in ToF
times a bivariate Gaussian on the detector face, with the shape measured only
where the counts determine it and reused elsewhere, as profile_1d_profilelib
does for the line shape alone (after Mantid's IntegratePeaksProfileFitting).

Model, for a peak whose line shape has its centroid at time t0 and whose centre
on the detector at that time is (x0, y0):

  m(x, y, t) = A ibix(t; alpha, beta, sigma, T_ph) bvg(x - xc(t), y - yc(t))
  xc(t) = x0 + drift_x (t - t0),  yc(t) = y0 + drift_y (t - t0)
  T_ph  = t0 - (1/beta - 1/alpha)

The position is carried as the centroid t0 rather than as T_ph, which is not
where the peak is and moves with the shape (see ibix_centroid_offset), so that
positions of reflections with different shapes can be compared. The profile is
normalised to unit sum over the bounding box, so A is the intensity inside the
box: the quantity summation measures.

Two fits:

  shape: every parameter free. Run on strong reflections to fill the library.
  forced: the shape held at a library entry's, and only (x0, y0, t0) free,
          with the amplitude solved linearly. Run on every reflection.

What differs from profile_3d_ibix, apart from the library:

  - ToF in us rather than in frames, as profile_1d_profilelib, so a library
    entry carries the same line shape across reflections whose ToF bins
    differ in width.
  - The peak position on the detector is a parameter rather than the brightest
    voxel, which on a weak reflection is a noise spike.
  - Bounds by reparameterisation, not by clamping (see bounded_to_internal).
  - Weights from the counts the model expects rather than the counts observed.
    Observed counts are mostly zero in the tails of a weak peak, and weighting
    by them discards the voxels that say the peak is not there, which biases
    the amplitude low. Solved by iterative reweighting, which converges to the
    Poisson maximum-likelihood solution.
  - The amplitude is not constrained to be positive: a weak reflection's
    intensity can come out negative, as summation's can, rather than being
    read high.
  - The variance is that of the weighted fit, inflated by the reduced
    chi-squared over the peak where the model does not describe the data, plus
    the error of the background level, which shifts every voxel of the box
    together.
*/

namespace dials { namespace algorithms {

  struct TOFProfile3DProfilelibParams {
    double alpha, alpha_min, alpha_max;  // line shape rise (1/us)
    double beta, beta_min, beta_max;     // line shape decay (1/us)
    double SigX_min, SigX_max;           // detector face widths (px)
    double SigY_min, SigY_max;
    double SigP_min, SigP_max;  // detector face correlation
    double max_drift_factor;  // bound on drift: box extent over ToF extent, times this
    int n_restarts;           // further attempts at a shape fit that fails
    bool show_profile_failures;
    double library_min_i_sigma;  // summation I/sigma a library reflection must reach
    double library_min_corr;     // correlation over the peak its shape fit must reach
    double position_tolerance;   // forced fit: scatters of library offsets allowed
    double free_position_min_i_sigma;  // I/sigma above which the position is fitted

    TOFProfile3DProfilelibParams(double alpha_,
                                 double alpha_min_,
                                 double alpha_max_,
                                 double beta_,
                                 double beta_min_,
                                 double beta_max_,
                                 double SigX_min_,
                                 double SigX_max_,
                                 double SigY_min_,
                                 double SigY_max_,
                                 double SigP_min_,
                                 double SigP_max_,
                                 double max_drift_factor_,
                                 int n_restarts_,
                                 bool show_profile_failures_,
                                 double library_min_i_sigma_,
                                 double library_min_corr_,
                                 double position_tolerance_,
                                 double free_position_min_i_sigma_)
        : alpha(alpha_),
          alpha_min(alpha_min_),
          alpha_max(alpha_max_),
          beta(beta_),
          beta_min(beta_min_),
          beta_max(beta_max_),
          SigX_min(SigX_min_),
          SigX_max(SigX_max_),
          SigY_min(SigY_min_),
          SigY_max(SigY_max_),
          SigP_min(SigP_min_),
          SigP_max(SigP_max_),
          max_drift_factor(max_drift_factor_),
          n_restarts(n_restarts_),
          show_profile_failures(show_profile_failures_),
          library_min_i_sigma(library_min_i_sigma_),
          library_min_corr(library_min_corr_),
          position_tolerance(position_tolerance_),
          free_position_min_i_sigma(free_position_min_i_sigma_) {}
  };

  // A peak shape in 3D, as fitted on one reflection and reusable on another
  struct Profile3DShape {
    double alpha = 0.0;  // line shape (1/us, 1/us, us)
    double beta = 0.0;
    double sigma = 0.0;
    double sig_x = 0.0;  // detector face (px)
    double sig_y = 0.0;
    double sig_p = 0.0;
    double drift_x = 0.0;  // movement of the centre with ToF (px/us)
    double drift_y = 0.0;
    bool valid = false;

    // Centroid of the line shape relative to T_ph (us)
    double centroid_offset() const {
      return 1.0 / beta - 1.0 / alpha;
    }
  };

  // Where a peak is: its centre on the detector (px) at the centroid of its
  // line shape (us)
  struct Profile3DPosition {
    double x = 0.0;
    double y = 0.0;
    double t = 0.0;
  };

  /*
   * The voxels of one bounding box, as a profile fit needs them. Indexed
   * (ix * ny + iy) * nz + iz.
   */
  struct Profile3DData {
    std::size_t nx = 0, ny = 0, nz = 0;
    std::vector<double> x;  // pixel centres (px)
    std::vector<double> y;
    std::vector<double> t;      // ToF of each slice (us)
    std::vector<double> value;  // corrected, background subtracted
    // Corrected value of one raw count. 0 marks a voxel the fit must not use.
    std::vector<double> gain;
    std::vector<double> background;      // background (raw counts)
    std::vector<double> extra_variance;  // from the incident and empty spectra
    // Variance of the background level (raw counts), an error shared by every
    // voxel of the box
    double background_level_variance = 0.0;

    std::size_t size() const {
      return nx * ny * nz;
    }
    std::size_t index(std::size_t ix, std::size_t iy, std::size_t iz) const {
      return (ix * ny + iy) * nz + iz;
    }
    bool usable(std::size_t i) const {
      return gain[i] > 0.0 && std::isfinite(gain[i]) && std::isfinite(value[i]);
    }
    bool consistent() const {
      std::size_t n = size();
      return n > 0 && x.size() == nx && y.size() == ny && t.size() == nz
             && value.size() == n && gain.size() == n && background.size() == n
             && extra_variance.size() == n;
    }
  };

  struct Profile3DFitResult {
    bool success = false;
    Profile3DShape shape;
    Profile3DPosition position;
    double intensity = 0.0;
    double variance = 0.0;
    double correlation = 0.0;     // model against data over the peak
    double chi2 = -1.0;           // reduced, over the peak; -1 if not measurable
    std::vector<double> profile;  // unit sum over the box
  };

  namespace profile_3d_profilelib_detail {

    // Expected raw counts per voxel below which a variance is not allowed to
    // fall, so that a voxel the model says is empty keeps a finite weight
    const double min_expected_counts = 0.01;

    // A voxel is part of the peak where the profile reaches this fraction of
    // its maximum
    const double peak_fraction = 0.05;

    /*
     * The unit-sum profile of a shape at a position, over the box. False if
     * the profile has no weight inside the box.
     */
    inline bool model(const Profile3DData& d,
                      const Profile3DShape& s,
                      const Profile3DPosition& p,
                      std::vector<double>& out) {
      out.assign(d.size(), 0.0);
      if (!(s.alpha > 0.0) || !(s.beta > 0.0) || !(s.sigma > 0.0) || !(s.sig_x > 0.0)
          || !(s.sig_y > 0.0)) {
        return false;
      }
      const double T_ph = p.t - s.centroid_offset();
      scitbx::af::shared<double> line =
        profile1d_func(scitbx::af::const_ref<double>(d.t.data(), d.t.size()),
                       1.0,
                       s.alpha,
                       s.beta,
                       s.sigma,
                       T_ph);
      for (std::size_t iz = 0; iz < d.nz; ++iz) {
        if (!std::isfinite(line[iz]) || line[iz] < 0.0) {
          line[iz] = 0.0;
        }
      }

      const bool drifts = s.drift_x != 0.0 || s.drift_y != 0.0;
      std::vector<double> face(d.nx * d.ny, 0.0);
      double total = 0.0;
      for (std::size_t iz = 0; iz < d.nz; ++iz) {
        if (iz == 0 || drifts) {
          double cx = p.x + s.drift_x * (d.t[iz] - p.t);
          double cy = p.y + s.drift_y * (d.t[iz] - p.t);
          for (std::size_t ix = 0; ix < d.nx; ++ix) {
            for (std::size_t iy = 0; iy < d.ny; ++iy) {
              face[ix * d.ny + iy] =
                bvg_func(d.x[ix] - cx, d.y[iy] - cy, s.sig_x, s.sig_y, s.sig_p);
            }
          }
        }
        if (line[iz] == 0.0) {
          continue;
        }
        for (std::size_t ix = 0; ix < d.nx; ++ix) {
          for (std::size_t iy = 0; iy < d.ny; ++iy) {
            double v = line[iz] * face[ix * d.ny + iy];
            out[d.index(ix, iy, iz)] = v;
            total += v;
          }
        }
      }
      if (!(total > 0.0) || !std::isfinite(total)) {
        return false;
      }
      for (auto& v : out) {
        v /= total;
      }
      return true;
    }

    // Variance of each voxel if every voxel held the same number of counts:
    // an unweighted fit, to start the reweighting from
    inline void uniform_variance(const Profile3DData& d,
                                 std::vector<double>& variance) {
      double counts = 0.0;
      std::size_t n = 0;
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (d.usable(i)) {
          counts += std::abs(d.value[i]) / d.gain[i] + std::max(d.background[i], 0.0);
          ++n;
        }
      }
      double mean = n > 0 ? std::max(counts / n, min_expected_counts) : 1.0;
      variance.assign(d.size(), 0.0);
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (d.usable(i)) {
          variance[i] = d.gain[i] * d.gain[i] * mean + d.extra_variance[i];
        }
      }
    }

    // Variance of each voxel from the counts the model expects there
    inline void model_variance(const Profile3DData& d,
                               const std::vector<double>& profile,
                               double amplitude,
                               std::vector<double>& variance) {
      variance.assign(d.size(), 0.0);
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (!d.usable(i)) {
          continue;
        }
        double expected = std::max(d.background[i], 0.0)
                          + std::max(amplitude * profile[i], 0.0) / d.gain[i];
        expected = std::max(expected, min_expected_counts);
        variance[i] = d.gain[i] * d.gain[i] * expected + d.extra_variance[i];
      }
    }

    // Weighted least-squares amplitude of a profile, and its information
    // sum(p^2 / v)
    inline bool solve_amplitude(const Profile3DData& d,
                                const std::vector<double>& profile,
                                const std::vector<double>& variance,
                                double& amplitude,
                                double& information) {
      double num = 0.0;
      information = 0.0;
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (!d.usable(i) || !(variance[i] > 0.0)) {
          continue;
        }
        num += profile[i] * d.value[i] / variance[i];
        information += profile[i] * profile[i] / variance[i];
      }
      if (!(information > 0.0) || !std::isfinite(num)) {
        return false;
      }
      amplitude = num / information;
      return std::isfinite(amplitude);
    }

    /*
     * The parameters a fit varies, carried internally unbounded (see
     * bounded_to_internal). With free_shape, in order: alpha, beta, sigma,
     * sig_x, sig_y, sig_p, drift_x, drift_y, x, y, t. Otherwise x, y, t only,
     * with the rest of the shape as given.
     */
    struct Parameterisation {
      bool free_shape = false;
      Profile3DShape fixed_shape;
      std::vector<double> lo, hi;

      int size() const {
        return free_shape ? 11 : 3;
      }

      void decode(const Eigen::VectorXd& u,
                  Profile3DShape& shape,
                  Profile3DPosition& position) const {
        std::vector<double> p(size());
        for (int j = 0; j < size(); ++j) {
          p[j] = internal_to_bounded(u[j], lo[j], hi[j]);
        }
        shape = fixed_shape;
        int k = 0;
        if (free_shape) {
          shape.alpha = p[0];
          shape.beta = p[1];
          shape.sigma = p[2];
          shape.sig_x = p[3];
          shape.sig_y = p[4];
          shape.sig_p = p[5];
          shape.drift_x = p[6];
          shape.drift_y = p[7];
          k = 8;
        }
        position.x = p[k];
        position.y = p[k + 1];
        position.t = p[k + 2];
      }

      Eigen::VectorXd encode(const Profile3DShape& shape,
                             const Profile3DPosition& position) const {
        std::vector<double> p;
        if (free_shape) {
          p = {shape.alpha,
               shape.beta,
               shape.sigma,
               shape.sig_x,
               shape.sig_y,
               shape.sig_p,
               shape.drift_x,
               shape.drift_y};
        }
        p.push_back(position.x);
        p.push_back(position.y);
        p.push_back(position.t);
        Eigen::VectorXd u(size());
        for (int j = 0; j < size(); ++j) {
          u[j] = bounded_to_internal(p[j], lo[j], hi[j]);
        }
        return u;
      }
    };

    /*
     * Residuals (d - A p) / sqrt(v) over the usable voxels, with A solved
     * linearly for every trial shape and position, and v held fixed for the
     * duration of one minimisation.
     */
    struct Functor {
      const Profile3DData& data;
      const Parameterisation& parameterisation;
      const std::vector<double>& variance;
      std::vector<std::size_t> used;

      Functor(const Profile3DData& data_,
              const Parameterisation& parameterisation_,
              const std::vector<double>& variance_)
          : data(data_), parameterisation(parameterisation_), variance(variance_) {
        for (std::size_t i = 0; i < data.size(); ++i) {
          if (data.usable(i) && variance[i] > 0.0) {
            used.push_back(i);
          }
        }
      }

      int inputs() const {
        return parameterisation.size();
      }
      int values() const {
        return static_cast<int>(used.size());
      }

      int operator()(const Eigen::VectorXd& u, Eigen::VectorXd& fvec) const {
        Profile3DShape shape;
        Profile3DPosition position;
        parameterisation.decode(u, shape, position);
        std::vector<double> profile;
        double amplitude = 0.0, information = 0.0;
        bool ok = model(data, shape, position, profile)
                  && solve_amplitude(data, profile, variance, amplitude, information);
        fvec.resize(used.size());
        for (std::size_t k = 0; k < used.size(); ++k) {
          std::size_t i = used[k];
          double m = ok ? amplitude * profile[i] : 0.0;
          double r = (data.value[i] - m) / std::sqrt(variance[i]);
          fvec[k] = std::isfinite(r) ? r : 1e6;
        }
        return 0;
      }

      int df(const Eigen::VectorXd& u, Eigen::MatrixXd& J) const {
        J.resize(used.size(), inputs());
        Eigen::VectorXd f0(used.size());
        operator()(u, f0);
        for (int j = 0; j < inputs(); ++j) {
          Eigen::VectorXd up = u;
          double delta = 1e-5 * std::max(1.0, std::abs(u[j]));
          up[j] += delta;
          Eigen::VectorXd fp(used.size());
          operator()(up, fp);
          J.col(j) = (fp - f0) / delta;
        }
        return 0;
      }
    };

    inline double median(std::vector<double> values) {
      if (values.empty()) {
        return 0.0;
      }
      std::size_t mid = values.size() / 2;
      std::nth_element(values.begin(), values.begin() + mid, values.end());
      double upper = values[mid];
      if (values.size() % 2 == 1) {
        return upper;
      }
      double lower = *std::max_element(values.begin(), values.begin() + mid);
      return 0.5 * (lower + upper);
    }

    // 1.4826 times the median absolute deviation from the median
    inline double robust_sd(const std::vector<double>& values) {
      double centre = median(values);
      std::vector<double> deviations(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        deviations[i] = std::abs(values[i] - centre);
      }
      return 1.4826 * median(deviations);
    }

    inline bool summarise(const Profile3DData& d,
                          const std::vector<double>& variance,
                          std::size_t n_par,
                          Profile3DFitResult& result);

    /*
     * Minimises from u, reweighting from the model between minimisations.
     * Fills the result from the last of them.
     */
    inline bool minimise(const Profile3DData& d,
                         const Parameterisation& parameterisation,
                         Eigen::VectorXd& u,
                         int n_reweight,
                         int maxfev,
                         Profile3DFitResult& result) {
      std::vector<double> variance;
      uniform_variance(d, variance);
      for (int round = 0; round <= n_reweight; ++round) {
        Functor functor(d, parameterisation, variance);
        if (functor.values() <= functor.inputs()) {
          return false;
        }
        Eigen::LevenbergMarquardt<Functor, double> lm(functor);
        lm.parameters.maxfev = maxfev;
        lm.parameters.xtol = 1e-8;
        lm.parameters.ftol = 1e-8;
        int status = lm.minimize(u);
        if (status <= 0 || !u.allFinite()) {
          return false;
        }
        parameterisation.decode(u, result.shape, result.position);
        double amplitude = 0.0, information = 0.0;
        if (!model(d, result.shape, result.position, result.profile)
            || !solve_amplitude(d, result.profile, variance, amplitude, information)) {
          return false;
        }
        model_variance(d, result.profile, amplitude, variance);
      }
      return summarise(d, variance, parameterisation.size() + 1, result);
    }

    /*
     * The amplitude of result.profile under the given weights, and what the
     * fit says about it: the correlation and reduced chi-squared over the
     * peak, and the variance. n_par counts the parameters fitted, amplitude
     * included.
     */
    inline bool summarise(const Profile3DData& d,
                          const std::vector<double>& variance,
                          std::size_t n_par,
                          Profile3DFitResult& result) {
      double amplitude = 0.0, information = 0.0;
      if (!solve_amplitude(d, result.profile, variance, amplitude, information)) {
        return false;
      }

      double peak_max = 0.0;
      for (std::size_t i = 0; i < d.size(); ++i) {
        peak_max = std::max(peak_max, result.profile[i]);
      }
      double chi2 = 0.0;
      std::size_t n_peak = 0;
      double sd = 0.0, sm = 0.0, sdd = 0.0, smm = 0.0, sdm = 0.0;
      double shift = 0.0;  // d A / d (background level), times -information
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (!d.usable(i) || !(variance[i] > 0.0)) {
          continue;
        }
        shift += result.profile[i] * d.gain[i] / variance[i];
        if (result.profile[i] < peak_fraction * peak_max) {
          continue;
        }
        double m = amplitude * result.profile[i];
        double r = d.value[i] - m;
        chi2 += r * r / variance[i];
        ++n_peak;
        sd += d.value[i];
        sm += m;
        sdd += d.value[i] * d.value[i];
        smm += m * m;
        sdm += d.value[i] * m;
      }
      if (n_peak > 2) {
        double cov = sdm - sd * sm / n_peak;
        double var_d = sdd - sd * sd / n_peak;
        double var_m = smm - sm * sm / n_peak;
        result.correlation =
          (var_d > 0.0 && var_m > 0.0) ? cov / std::sqrt(var_d * var_m) : 0.0;
      }
      double inflation = 1.0;
      if (n_peak > n_par) {
        result.chi2 = chi2 / (n_peak - n_par);
        inflation = std::max(result.chi2, 1.0);
      }

      double slope = shift / information;
      result.intensity = amplitude;
      result.variance =
        inflation / information + d.background_level_variance * slope * slope;
      result.success = std::isfinite(result.intensity) && std::isfinite(result.variance)
                       && result.variance > 0.0;
      return result.success;
    }

    // Starting point for a shape fit, from moments of the data
    struct ShapeSeed {
      Profile3DShape shape;
      Profile3DPosition position;
    };

    inline bool seed_shape(const Profile3DData& d,
                           const TOFProfile3DProfilelibParams& params,
                           ShapeSeed& seed,
                           Parameterisation& parameterisation) {
      // ToF projection of the positive part
      std::vector<double> projection(d.nz, 0.0);
      for (std::size_t ix = 0; ix < d.nx; ++ix) {
        for (std::size_t iy = 0; iy < d.ny; ++iy) {
          for (std::size_t iz = 0; iz < d.nz; ++iz) {
            std::size_t i = d.index(ix, iy, iz);
            if (d.usable(i) && d.value[i] > 0.0) {
              projection[iz] += d.value[i];
            }
          }
        }
      }
      // Lightly smoothed, so the peak is not a single noisy slice
      std::vector<double> smooth(d.nz, 0.0);
      for (std::size_t iz = 0; iz < d.nz; ++iz) {
        double total = 0.0;
        int n = 0;
        for (int k = -1; k <= 1; ++k) {
          int j = static_cast<int>(iz) + k;
          if (j >= 0 && j < static_cast<int>(d.nz)) {
            total += projection[j];
            ++n;
          }
        }
        smooth[iz] = total / n;
      }
      std::size_t iz_max =
        std::distance(smooth.begin(), std::max_element(smooth.begin(), smooth.end()));
      double peak = smooth[iz_max];
      if (!(peak > 0.0)) {
        return false;
      }
      const double mean_dt = (d.t.back() - d.t.front()) / std::max<double>(d.nz - 1, 1);
      if (!(mean_dt > 0.0)) {
        return false;
      }

      // Width in ToF from the full width at half maximum
      double half = 0.5 * peak;
      std::size_t lo = iz_max, hi = iz_max;
      while (lo > 0 && smooth[lo - 1] > half)
        --lo;
      while (hi + 1 < d.nz && smooth[hi + 1] > half)
        ++hi;
      double fwhm = d.t[hi] - d.t[lo] + mean_dt;
      double sigma0 = std::max(fwhm / 2.354820045, mean_dt);

      // Detector face moments over the slices holding the peak
      std::vector<double> image(d.nx * d.ny, 0.0);
      for (std::size_t iz = 0; iz < d.nz; ++iz) {
        if (smooth[iz] < 0.1 * peak) {
          continue;
        }
        for (std::size_t ix = 0; ix < d.nx; ++ix) {
          for (std::size_t iy = 0; iy < d.ny; ++iy) {
            std::size_t i = d.index(ix, iy, iz);
            if (d.usable(i) && d.value[i] > 0.0) {
              image[ix * d.ny + iy] += d.value[i];
            }
          }
        }
      }
      double s0 = 0.0, sx = 0.0, sy = 0.0;
      for (std::size_t ix = 0; ix < d.nx; ++ix) {
        for (std::size_t iy = 0; iy < d.ny; ++iy) {
          double v = image[ix * d.ny + iy];
          s0 += v;
          sx += v * d.x[ix];
          sy += v * d.y[iy];
        }
      }
      if (!(s0 > 0.0)) {
        return false;
      }
      double cx = sx / s0, cy = sy / s0;
      double mxx = 0.0, myy = 0.0, mxy = 0.0;
      for (std::size_t ix = 0; ix < d.nx; ++ix) {
        for (std::size_t iy = 0; iy < d.ny; ++iy) {
          double v = image[ix * d.ny + iy];
          double dx = d.x[ix] - cx, dy = d.y[iy] - cy;
          mxx += v * dx * dx;
          myy += v * dy * dy;
          mxy += v * dx * dy;
        }
      }
      mxx /= s0;
      myy /= s0;
      mxy /= s0;

      auto clamp_inside = [](double v, double a, double b) {
        double margin = 1e-3 * (b - a);
        return std::min(std::max(v, a + margin), b - margin);
      };

      Profile3DShape& s = seed.shape;
      s.alpha = params.alpha;
      s.beta = params.beta;
      s.sigma = sigma0;
      s.sig_x = std::sqrt(std::max(mxx, 0.0));
      s.sig_y = std::sqrt(std::max(myy, 0.0));
      s.sig_p = (s.sig_x > 0.0 && s.sig_y > 0.0) ? mxy / (s.sig_x * s.sig_y) : 0.0;
      s.drift_x = 0.0;
      s.drift_y = 0.0;
      s.valid = true;
      seed.position.x = cx;
      seed.position.y = cy;
      seed.position.t = d.t[iz_max];

      const double tof_range = d.t.back() - d.t.front() + mean_dt;
      const double drift_x_max = params.max_drift_factor * d.nx / tof_range;
      const double drift_y_max = params.max_drift_factor * d.ny / tof_range;
      parameterisation.free_shape = true;
      parameterisation.lo = {params.alpha_min,
                             params.beta_min,
                             sigma0 / 4.0,
                             params.SigX_min,
                             params.SigY_min,
                             params.SigP_min,
                             -drift_x_max,
                             -drift_y_max,
                             d.x.front() - 0.5,
                             d.y.front() - 0.5,
                             d.t.front()};
      parameterisation.hi = {params.alpha_max,
                             params.beta_max,
                             std::max(100.0, 4.0 * sigma0),
                             params.SigX_max,
                             params.SigY_max,
                             params.SigP_max,
                             drift_x_max,
                             drift_y_max,
                             d.x.back() + 0.5,
                             d.y.back() + 0.5,
                             d.t.back()};
      for (std::size_t j = 0; j < parameterisation.lo.size(); ++j) {
        if (!(parameterisation.hi[j] > parameterisation.lo[j])) {
          return false;
        }
      }
      const std::vector<double>& plo = parameterisation.lo;
      const std::vector<double>& phi = parameterisation.hi;
      s.alpha = clamp_inside(s.alpha, plo[0], phi[0]);
      s.beta = clamp_inside(s.beta, plo[1], phi[1]);
      s.sigma = clamp_inside(s.sigma, plo[2], phi[2]);
      s.sig_x = clamp_inside(s.sig_x, plo[3], phi[3]);
      s.sig_y = clamp_inside(s.sig_y, plo[4], phi[4]);
      s.sig_p = clamp_inside(s.sig_p, plo[5], phi[5]);
      seed.position.x = clamp_inside(seed.position.x, plo[8], phi[8]);
      seed.position.y = clamp_inside(seed.position.y, plo[9], phi[9]);
      seed.position.t = clamp_inside(seed.position.t, plo[10], phi[10]);
      return true;
    }

    // Whether a parameter sits against a bound, where the data did not decide it
    inline bool at_bound(double v, double lo, double hi) {
      double margin = 1e-3 * (hi - lo);
      return v <= lo + margin || v >= hi - margin;
    }

  }  // namespace profile_3d_profilelib_detail

  /*
   * Fit every parameter of the model to one reflection, for the library.
   *
   * Succeeds only where the result is a shape worth reusing: the model
   * correlates with the data over the peak at least min_correlation, the
   * amplitude is positive, and neither width on the detector nor the
   * correlation between them has run to a bound.
   */
  inline Profile3DFitResult fit_profile_3d_profilelib_shape(
    const Profile3DData& d,
    const TOFProfile3DProfilelibParams& params,
    double min_correlation,
    int n_reweight = 2) {
    namespace detail = profile_3d_profilelib_detail;
    Profile3DFitResult best;
    if (!d.consistent() || d.nx < 3 || d.ny < 3 || d.nz < 5) {
      return best;
    }
    detail::ShapeSeed seed;
    detail::Parameterisation parameterisation;
    if (!detail::seed_shape(d, params, seed, parameterisation)) {
      return best;
    }
    const std::vector<double>& lo = parameterisation.lo;
    const std::vector<double>& hi = parameterisation.hi;

    auto acceptable = [&](const Profile3DFitResult& r, const char*& why) {
      if (!r.success) {
        why = "fit failed";
        return false;
      }
      if (!(r.intensity > 0.0)) {
        why = "intensity not positive";
        return false;
      }
      if (!(r.correlation >= min_correlation)) {
        why = "low correlation";
        return false;
      }
      if (detail::at_bound(r.shape.sig_x, lo[3], hi[3])
          || detail::at_bound(r.shape.sig_y, lo[4], hi[4])
          || detail::at_bound(r.shape.sig_p, lo[5], hi[5])) {
        why = "detector face shape at a bound";
        return false;
      }
      return true;
    };

    const Eigen::VectorXd u0 = parameterisation.encode(seed.shape, seed.position);
    std::mt19937 rng(12345);
    std::normal_distribution<double> noise(0.0, 0.5);
    const char* why = "";
    for (int attempt = 0; attempt <= params.n_restarts; ++attempt) {
      Eigen::VectorXd u = u0;
      if (attempt > 0) {
        double scale = attempt <= params.n_restarts / 2 ? 0.5 : 1.5;
        for (int j = 0; j < u.size(); ++j) {
          u[j] += scale * noise(rng);
        }
      }
      Profile3DFitResult result;
      detail::minimise(d, parameterisation, u, n_reweight, 300, result);
      if (acceptable(result, why)) {
        result.shape.valid = true;
        return result;
      }
    }
    if (params.show_profile_failures) {
      std::cerr << "profile_3d_profilelib shape fit failure: " << why << "\n";
    }
    return best;
  }

  /*
   * Fit one reflection with the shape held fixed: only the position and the
   * amplitude are free.
   *
   * The position starts at seed and is kept within tolerance of it (px, px,
   * us), and inside the box. Without a seed it starts at the centre of the box
   * on the detector and at the brightest ToF slice, and may go anywhere in the
   * box.
   */
  inline Profile3DFitResult fit_profile_3d_profilelib_forced(
    const Profile3DData& d,
    const Profile3DShape& shape,
    const Profile3DPosition* seed = nullptr,
    const Profile3DPosition* tolerance = nullptr,
    int n_reweight = 3) {
    namespace detail = profile_3d_profilelib_detail;
    Profile3DFitResult result;
    if (!d.consistent() || !shape.valid || d.nz < 2) {
      return result;
    }
    const double mean_dt = (d.t.back() - d.t.front()) / std::max<double>(d.nz - 1, 1);
    if (!(mean_dt > 0.0)) {
      return result;
    }

    detail::Parameterisation parameterisation;
    parameterisation.free_shape = false;
    parameterisation.fixed_shape = shape;
    std::vector<double> box_lo = {d.x.front() - 0.5, d.y.front() - 0.5, d.t.front()};
    std::vector<double> box_hi = {d.x.back() + 0.5, d.y.back() + 0.5, d.t.back()};

    Profile3DPosition start;
    if (seed != nullptr) {
      start = *seed;
    } else {
      start.x = 0.5 * (box_lo[0] + box_hi[0]);
      start.y = 0.5 * (box_lo[1] + box_hi[1]);
      std::vector<double> projection(d.nz, 0.0);
      for (std::size_t i = 0; i < d.size(); ++i) {
        if (d.usable(i)) {
          projection[i % d.nz] += d.value[i];
        }
      }
      std::vector<double> smooth(d.nz, 0.0);
      for (std::size_t iz = 0; iz < d.nz; ++iz) {
        double total = 0.0;
        int n = 0;
        for (int k = -1; k <= 1; ++k) {
          int j = static_cast<int>(iz) + k;
          if (j >= 0 && j < static_cast<int>(d.nz)) {
            total += projection[j];
            ++n;
          }
        }
        smooth[iz] = total / n;
      }
      start.t = d.t[std::distance(smooth.begin(),
                                  std::max_element(smooth.begin(), smooth.end()))];
    }

    double start_v[3] = {start.x, start.y, start.t};
    parameterisation.lo.resize(3);
    parameterisation.hi.resize(3);
    for (int j = 0; j < 3; ++j) {
      double lo = box_lo[j], hi = box_hi[j];
      if (tolerance != nullptr) {
        double tol[3] = {tolerance->x, tolerance->y, tolerance->t};
        lo = std::max(lo, start_v[j] - tol[j]);
        hi = std::min(hi, start_v[j] + tol[j]);
      }
      if (!(hi > lo)) {
        // The seed lies outside the box, or the tolerance leaves no room
        return result;
      }
      parameterisation.lo[j] = lo;
      parameterisation.hi[j] = hi;
      double margin = 1e-3 * (hi - lo);
      start_v[j] = std::min(std::max(start_v[j], lo + margin), hi - margin);
    }
    start.x = start_v[0];
    start.y = start_v[1];
    start.t = start_v[2];

    Eigen::VectorXd u = parameterisation.encode(shape, start);
    detail::minimise(d, parameterisation, u, n_reweight, 200, result);
    return result;
  }

  /*
   * The shape broadened by an uncertain position: a peak at its expected
   * position plus an error of standard deviation scatter (px, px, us),
   * averaged over that error, is its own shape convolved with the
   * distribution of the error. For a Gaussian error that adds variances, on
   * the detector and in ToF alike, where the line shape already is a function
   * convolved with a Gaussian.
   */
  inline Profile3DShape broaden_profile_3d_shape(const Profile3DShape& shape,
                                                 const Profile3DPosition& scatter) {
    Profile3DShape out = shape;
    out.sigma = std::sqrt(shape.sigma * shape.sigma + scatter.t * scatter.t);
    out.sig_x = std::sqrt(shape.sig_x * shape.sig_x + scatter.x * scatter.x);
    out.sig_y = std::sqrt(shape.sig_y * shape.sig_y + scatter.y * scatter.y);
    out.sig_p = shape.sig_p * shape.sig_x * shape.sig_y / (out.sig_x * out.sig_y);
    return out;
  }

  /*
   * Fit one reflection with the shape and the position both held fixed: only
   * the amplitude is free, so it is a linear fit, reweighted from the model.
   *
   * A position fitted to a weak peak goes wherever the noise is brightest,
   * which reads the intensity high and its error low. Held at its expected
   * place, with the shape broadened by how uncertain that place is (see
   * broaden_profile_3d_shape), the fit is unbiased on average.
   */
  inline Profile3DFitResult fit_profile_3d_profilelib_fixed(
    const Profile3DData& d,
    const Profile3DShape& shape,
    const Profile3DPosition& position,
    int n_reweight = 3) {
    namespace detail = profile_3d_profilelib_detail;
    Profile3DFitResult result;
    if (!d.consistent() || !shape.valid) {
      return result;
    }
    result.shape = shape;
    result.position = position;
    if (!detail::model(d, shape, position, result.profile)) {
      return result;
    }
    std::vector<double> variance;
    detail::uniform_variance(d, variance);
    for (int round = 0; round <= n_reweight; ++round) {
      double amplitude = 0.0, information = 0.0;
      if (!detail::solve_amplitude(
            d, result.profile, variance, amplitude, information)) {
        return result;
      }
      detail::model_variance(d, result.profile, amplitude, variance);
    }
    detail::summarise(d, variance, 1, result);
    return result;
  }

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_PROFILELIB_H
