#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_LINE_SHAPE_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_LINE_SHAPE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include "tof_profile_3d_bvg.h"
#include "tof_utils.h"

/*
 * The line shape of a peak in time of flight: the moderator's pulse, as it
 * reaches the detector, for one of the models the profile methods fit.
 *
 * iBIX: back-to-back exponentials (rise alpha, decay beta) convolved with a
 * Gaussian (sigma). Yano, N., Yamada, T., Hosoya, T. et al., Sci Rep 6, 36628
 * (2016). https://doi.org/10.1038/srep36628
 *
 * Ikeda-Carpenter: the fast (A) and slow (B) moderator components in ratio R,
 * convolved with a rectangle (half-width hat) and a Gaussian (sigma). Ikeda,
 * S.I., Carpenter, J.M., Nucl. Instrum. Methods Phys. Res. A 239(3), 536-544
 * (1985). https://doi.org/10.1016/0168-9002(85)90033-6. The convolutions are
 * discrete, over the slices of the box, with the widths in us converted to
 * slices at the box's mean slice width.
 *
 * Every parameter is in us or 1/us, so a shape measured on one reflection
 * applies to another whose slices differ in width. A peak is placed by a
 * reference time that is comparable between shapes: the centroid for iBIX,
 * the mode of the unconvolved pulse for Ikeda-Carpenter, whose slow tail
 * makes its centroid depend on B far more than where the peak is does.
 */

namespace dials { namespace algorithms {

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

  template <typename T>
  static scitbx::af::shared<T> ic_raw_t(scitbx::af::const_ref<double> tof,
                                        T A,
                                        T B,
                                        T R,
                                        T T0) {
    /*
     * Ikeda-Carpenter function with unit amplitude for each tof[i].
     */

    const int n = tof.size();
    scitbx::af::shared<T> out(n, T(0.0));
    T dAB = A - B;
    if (std::abs(as_real(dAB)) < 1e-8) dAB = T(as_real(dAB) >= 0.0 ? 1e-8 : -1e-8);
    const T dAB3 = dAB * dAB * dAB;
    const T coeff2 = T(2.0) * R * A * A * B / dAB3;
    for (int i = 0; i < n; ++i) {
      const T dt = T(tof[i]) - T0;
      if (as_real(dt) <= 0.0) continue;
      const T Adt = A * dt;
      const T Bdt = B * dt;
      const T dABdt = dAB * dt;
      const T expA = exp_safe_t(-Adt);
      const T expB = exp_safe_t(-Bdt);
      const T term1 = (T(1.0) - R) * Adt * Adt * expA;
      const T term2 =
        coeff2 * (expB - expA * (T(1.0) + dABdt + T(0.5) * dABdt * dABdt));
      const T val = term1 + term2;
      out[i] = (std::isfinite(as_real(val)) && as_real(val) > 0.0) ? val : T(0.0);
    }
    return out;
  }

  static scitbx::af::shared<double> ic_raw(scitbx::af::const_ref<double> tof,
                                           double A,
                                           double B,
                                           double R,
                                           double T0) {
    return ic_raw_t<double>(tof, A, B, R, T0);
  }

  static scitbx::af::shared<double> make_hat_kernel(int n, double hw) {
    // Normalised hat kernel of length n, half-width hw bins

    scitbx::af::shared<double> k(n, 0.0);
    const int mid = n / 2;
    const int lo = std::max(0, static_cast<int>(std::floor(mid - std::abs(hw))));
    const int hi = std::min(n, static_cast<int>(std::ceil(mid + std::abs(hw))));
    for (int i = lo; i < hi; ++i)
      k[i] = 1.0;
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
      sum += k[i];
    if (sum > 0.0)
      for (int i = 0; i < n; ++i)
        k[i] /= sum;
    else
      k[mid] = 1.0;  // identity kernel when hw=0
    return k;
  }

  static scitbx::af::shared<double> make_gauss_kernel(int n, double kconv) {
    /*
     * Normalised Gaussian kernel of length n.
     * kconv is a decay rate inToF-bin units
     */

    scitbx::af::shared<double> k(n, 0.0);
    const double mid = (n > 1) ? 0.5 * (n - 1) : 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      const double d = i - mid;
      k[i] = std::exp(-kconv * d * d);
      sum += k[i];
    }
    if (sum > 0.0)
      for (int i = 0; i < n; ++i)
        k[i] /= sum;
    return k;
  }

  static scitbx::af::shared<double> convolve(scitbx::af::const_ref<double> f,
                                             scitbx::af::const_ref<double> kernel) {
    // Discrete convolution

    const int n = f.size();
    const int m = kernel.size();
    const int first = n / 2;
    scitbx::af::shared<double> out(n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        const int ki = i + first - k;
        if (ki >= 0 && ki < m) out[i] += f[k] * kernel[ki];
      }
    }
    return out;
  }

  static scitbx::af::shared<double> ic_profile(scitbx::af::const_ref<double> tof,
                                               double A,
                                               double B,
                                               double R,
                                               double T0,
                                               double HatWidth,
                                               double KConv) {
    const int n = tof.size();
    scitbx::af::shared<double> f = ic_raw(tof, A, B, R, T0);
    scitbx::af::shared<double> hat = make_hat_kernel(n, HatWidth);
    scitbx::af::shared<double> gauss = make_gauss_kernel(n, KConv);
    scitbx::af::shared<double> f_hat = convolve(f.const_ref(), hat.const_ref());
    return convolve(f_hat.const_ref(), gauss.const_ref());
  }

  // The unconvolved Ikeda-Carpenter pulse at one time after its onset
  inline double ic_raw_value(double dt, double A, double B, double R) {
    // As ic_raw_t, for one time
    if (dt <= 0.0) {
      return 0.0;
    }
    double dAB = A - B;
    if (std::abs(dAB) < 1e-8) dAB = dAB >= 0.0 ? 1e-8 : -1e-8;
    const double coeff2 = 2.0 * R * A * A * B / (dAB * dAB * dAB);
    const double Adt = A * dt;
    const double dABdt = dAB * dt;
    const double expA = exp_safe_t(-Adt);
    const double expB = exp_safe_t(-B * dt);
    const double val = (1.0 - R) * Adt * Adt * expA
                       + coeff2 * (expB - expA * (1.0 + dABdt + 0.5 * dABdt * dABdt));
    return (std::isfinite(val) && val > 0.0) ? val : 0.0;
  }

  /*
   * Time from onset to the maximum of the unconvolved Ikeda-Carpenter pulse:
   * a coarse grid over 20/A, where the fast component has died away, then a
   * golden-section search about the best point.
   */
  inline double ic_mode_offset(double A, double B, double R) {
    if (!(A > 0.0) || !(B > 0.0)) {
      return 0.0;
    }
    const double span = 20.0 / A;
    const int n = 400;
    int best = 1;
    double best_v = -1.0;
    for (int i = 1; i <= n; ++i) {
      double v = ic_raw_value(span * i / n, A, B, R);
      if (v > best_v) {
        best_v = v;
        best = i;
      }
    }
    double lo = span * (best - 1) / n, hi = span * (best + 1) / n;
    const double g = 0.5 * (std::sqrt(5.0) - 1.0);
    double c = hi - g * (hi - lo), d = lo + g * (hi - lo);
    double fc = ic_raw_value(c, A, B, R), fd = ic_raw_value(d, A, B, R);
    for (int it = 0; it < 40; ++it) {
      if (fc > fd) {
        hi = d;
        d = c;
        fd = fc;
        c = hi - g * (hi - lo);
        fc = ic_raw_value(c, A, B, R);
      } else {
        lo = c;
        c = d;
        fc = fd;
        d = lo + g * (hi - lo);
        fd = ic_raw_value(d, A, B, R);
      }
    }
    return 0.5 * (lo + hi);
  }

  struct LineShape {
    enum Kind { IBIX = 0, IC = 1 };
    Kind kind = IBIX;
    /*
     * IBIX: alpha (1/us), beta (1/us), sigma (us).
     * IC: A (1/us), B (1/us), R, hat half-width (us), sigma (us).
     */
    std::array<double, 5> p = {{0.0, 0.0, 0.0, 0.0, 0.0}};

    static int n_params(Kind kind) {
      return kind == IBIX ? 3 : 5;
    }
    int n_params() const {
      return n_params(kind);
    }

    // Index of the Gaussian width, which position errors broaden
    static int width_index(Kind kind) {
      return kind == IBIX ? 2 : 4;
    }
    int width_index() const {
      return width_index(kind);
    }

    // Every rate and width positive (the hat may be zero)
    bool positive() const {
      if (kind == IBIX) {
        return p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0;
      }
      return p[0] > 0.0 && p[1] > 0.0 && p[2] >= 0.0 && p[2] <= 1.0 && p[3] >= 0.0
             && p[4] > 0.0;
    }

    /*
     * The reference time minus the onset parameter of the model (T_ph for
     * iBIX, T0 for Ikeda-Carpenter).
     */
    double reference_offset() const {
      if (kind == IBIX) {
        return 1.0 / p[1] - 1.0 / p[0];
      }
      return ic_mode_offset(p[0], p[1], p[2]);
    }

    /*
     * The line at times t, with the model's own amplitude parameter and onset.
     * For iBIX the amplitude multiplies a unit-area pulse.
     */
    scitbx::af::shared<double> evaluate_at_onset(scitbx::af::const_ref<double> t,
                                                 double amplitude,
                                                 double onset) const {
      if (kind == IBIX) {
        return profile1d_func(t, amplitude, p[0], p[1], p[2], onset);
      }
      const std::size_t n = t.size();
      const double dt = n > 1 ? (t[n - 1] - t[0]) / (n - 1) : 1.0;
      const double hat_bins = dt > 0.0 ? std::max(p[3] / dt, 0.0) : 0.0;
      // A Gaussian narrower than a twentieth of a slice is a delta function
      const double sigma_bins = dt > 0.0 ? std::max(p[4] / dt, 0.05) : 0.05;
      const double kconv = 1.0 / (2.0 * sigma_bins * sigma_bins);
      scitbx::af::shared<double> out =
        ic_profile(t, p[0], p[1], p[2], onset, hat_bins, kconv);
      if (amplitude != 1.0) {
        for (std::size_t i = 0; i < out.size(); ++i) {
          out[i] *= amplitude;
        }
      }
      return out;
    }

    // The line at times t for a peak whose reference time is t_ref
    scitbx::af::shared<double> evaluate(scitbx::af::const_ref<double> t,
                                        double t_ref) const {
      return evaluate_at_onset(t, 1.0, t_ref - reference_offset());
    }

    // Convolved with a Gaussian error of standard deviation s_t in the position
    LineShape broadened(double s_t) const {
      LineShape out(*this);
      const int w = width_index();
      out.p[w] = std::sqrt(p[w] * p[w] + s_t * s_t);
      return out;
    }

    static LineShape ibix(double alpha, double beta, double sigma) {
      LineShape s;
      s.kind = IBIX;
      s.p = {{alpha, beta, sigma, 0.0, 0.0}};
      return s;
    }
  };

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_LINE_SHAPE_H */
