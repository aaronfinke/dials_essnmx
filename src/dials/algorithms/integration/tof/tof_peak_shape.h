#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_PEAK_SHAPE_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_PEAK_SHAPE_H

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <dials/algorithms/integration/tof/tof_profile_1d_ibix.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_bvg.h>

/*
 * The shape of a peak in three dimensions, independent of how it was measured.
 *
 * Overlap deconvolution (tof_overlap_deconvolution.h) needs the profile of
 * each reflection of a cluster over its bounding box, and nothing else about
 * the model that produced it. PeakShape is that: a shape that can be placed
 * at a position, evaluated over a box and normalised there, and broadened by
 * how uncertain the position is. Each family of profile model implements it
 * once, and every method built on that family shares it.
 */

namespace dials { namespace algorithms {

  /*
   * Where a peak is: its centre on the detector (px) at the centroid of its
   * line shape in ToF (us). Also used for differences of positions (an offset
   * from the prediction, or the scatter of offsets).
   */
  struct PeakPosition {
    double x = 0.0;
    double y = 0.0;
    double t = 0.0;
  };

  /*
   * The centroid of the iBIX ToF line shape relative to its T_ph parameter.
   * The shape is a rising exponential (rate alpha) joined to a falling one
   * (rate beta), convolved with a Gaussian, whose mean is 1/beta - 1/alpha.
   * T_ph is therefore not where the peak is, and how far off it is depends on
   * the shape, so the position of a peak is carried as its centroid: that can
   * be compared between reflections of different shape.
   */
  inline double ibix_centroid_offset(const IBIXShape& shape) {
    return 1.0 / shape.beta - 1.0 / shape.alpha;
  }

  class PeakShape {
  public:
    virtual ~PeakShape() {}

    /*
     * The profile over a box of nx x ny pixels starting at pixel (x0, y0),
     * at the ToF of each of its slices, for a peak centred at centre.
     * Normalised to unit sum over the box. Written to profile with index
     * (ix * ny + iy) * nz + iz, nz = tof.size().
     *
     * Returns false if the profile holds nothing inside the box.
     */
    virtual bool box_profile(int x0,
                             int y0,
                             std::size_t nx,
                             std::size_t ny,
                             const std::vector<double>& tof,
                             const PeakPosition& centre,
                             std::vector<double>& profile) const = 0;

    /*
     * The expected profile of a peak whose true position scatters about the
     * one it is placed at, with standard deviations scatter: the shape
     * convolved with that scatter.
     */
    virtual std::shared_ptr<const PeakShape> broadened(
      const PeakPosition& scatter) const = 0;
  };

  /*
   * The iBIX line shape in ToF (back-to-back exponentials convolved with a
   * Gaussian) times a bivariate Gaussian on the detector face, whose centre
   * may move linearly with ToF. Used by profile_1d_ibix (with the face
   * measured separately and no drift) and profile_3d_ibix.
   */
  class IBIXGaussianShape : public PeakShape {
  public:
    IBIXShape line;      // T_ph unused: the position is given to box_profile
    double sig_x = 0.0;  // detector face (px)
    double sig_y = 0.0;
    double sig_p = 0.0;
    double drift_x = 0.0;  // movement of the centre with ToF (px/us), about
    double drift_y = 0.0;  // its position at the centroid of the line shape

    bool box_profile(int x0,
                     int y0,
                     std::size_t nx,
                     std::size_t ny,
                     const std::vector<double>& tof,
                     const PeakPosition& centre,
                     std::vector<double>& profile) const override {
      if (!line.valid) {
        return false;
      }
      const std::size_t nz = tof.size();

      scitbx::af::shared<double> tof_axis(nz);
      for (std::size_t z = 0; z < nz; ++z) {
        tof_axis[z] = tof[z];
      }
      const double t_ph = centre.t - ibix_centroid_offset(line);
      scitbx::af::shared<double> raw = profile1d_func(
        tof_axis.const_ref(), 1.0, line.alpha, line.beta, line.sigma, t_ph);
      std::vector<double> tof_profile(nz, 0.0);
      for (std::size_t z = 0; z < nz; ++z) {
        double v = raw[z];
        if (!std::isfinite(v) || v < 0.0) {
          v = 0.0;
        }
        tof_profile[z] = v;
      }

      // One face where the peak stays put, one per slice where it drifts
      const bool drifts = drift_x != 0.0 || drift_y != 0.0;
      const std::size_t n_faces = drifts ? nz : 1;
      std::vector<double> face(n_faces * nx * ny, 0.0);
      std::vector<double> face_sum(n_faces, 0.0);
      for (std::size_t f = 0; f < n_faces; ++f) {
        double dt = drifts ? tof[f] - centre.t : 0.0;
        double x_peak = centre.x + drift_x * dt;
        double y_peak = centre.y + drift_y * dt;
        for (std::size_t ix = 0; ix < nx; ++ix) {
          for (std::size_t iy = 0; iy < ny; ++iy) {
            double v = bvg_func(
              x0 + ix + 0.5 - x_peak, y0 + iy + 0.5 - y_peak, sig_x, sig_y, sig_p);
            face[f * nx * ny + ix * ny + iy] = v;
            face_sum[f] += v;
          }
        }
      }

      double norm = 0.0;
      for (std::size_t z = 0; z < nz; ++z) {
        norm += tof_profile[z] * face_sum[drifts ? z : 0];
      }
      if (!(norm > 0.0) || !std::isfinite(norm)) {
        return false;
      }

      profile.assign(nx * ny * nz, 0.0);
      for (std::size_t ix = 0; ix < nx; ++ix) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
          for (std::size_t z = 0; z < nz; ++z) {
            std::size_t f = drifts ? z : 0;
            profile[(ix * ny + iy) * nz + z] =
              tof_profile[z] * face[f * nx * ny + ix * ny + iy] / norm;
          }
        }
      }
      return true;
    }

    /*
     * For Gaussian scatter the convolution adds variances: in ToF too, where
     * the line shape is already a function convolved with a Gaussian. The
     * correlation on the face keeps its covariance.
     */
    IBIXGaussianShape broadened_copy(const PeakPosition& scatter) const {
      IBIXGaussianShape out(*this);
      out.line.sigma = std::sqrt(line.sigma * line.sigma + scatter.t * scatter.t);
      out.sig_x = std::sqrt(sig_x * sig_x + scatter.x * scatter.x);
      out.sig_y = std::sqrt(sig_y * sig_y + scatter.y * scatter.y);
      out.sig_p = sig_p * sig_x * sig_y / (out.sig_x * out.sig_y);
      return out;
    }

    std::shared_ptr<const PeakShape> broadened(
      const PeakPosition& scatter) const override {
      return std::make_shared<IBIXGaussianShape>(broadened_copy(scatter));
    }

    // Finite widths on every axis
    bool usable() const {
      return line.valid && std::isfinite(line.sigma) && sig_x > 0.0 && sig_y > 0.0;
    }
  };

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_PEAK_SHAPE_H */
