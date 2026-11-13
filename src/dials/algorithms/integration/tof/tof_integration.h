#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_INTEGRATION_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_INTEGRATION_H

#include <memory>
#include <boost/optional.hpp>
#include <dxtbx/imageset.h>
#include <dxtbx/format/image.h>
#include <dxtbx/array_family/flex_table.h>
#include <dials/model/data/shoebox.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/scan.h>
#include <dxtbx/model/goniometer.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/integration/processor.h>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/constants.h>
#include <dials/model/data/mask_code.h>
#include <dials/algorithms/integration/tof/tof_profile_1d_ibix.h>
#include <dials/algorithms/integration/tof/tof_profile_1d_profilelib.h>
#include <dials/algorithms/integration/tof/tof_profile_1d_ic.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_gutmann.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_ic.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_ibix.h>
#include <dials/algorithms/scaling/tof/tof_scaling.h>
#include <dials/util/thread_pool.h>
#include <atomic>
#include <map>
#include <chrono>
#include <mutex>
#include <vector>
#include <iostream>
#include <thread>

namespace dials { namespace algorithms {

  using dials::algorithms::Shoebox;
  using dials::algorithms::ShoeboxProcessor;
  using dials::model::Background;
  using dials::model::BackgroundUsed;
  using dials::model::Foreground;
  using dials::model::Overlapped;
  using dials::model::Valid;
  using dxtbx::ImageSequence;
  using dxtbx::af::flex_table;
  using dxtbx::model::Detector;
  using dxtbx::model::Experiment;
  using dxtbx::model::Goniometer;
  using dxtbx::model::PolychromaticBeam;
  using dxtbx::model::Scan;
  using dxtbx::model::scan_property_types;
  using scitbx::deg_as_rad;
  using scitbx::mat3;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::int6;
  using scitbx::constants::m_n;
  using scitbx::constants::pi;
  using scitbx::constants::Planck;

  // Holds information common to different shoeboxes
  struct TOFGeometryContext {
    Detector detector;
    vec3<double> unit_s0;
    double sample_to_source_distance;
    scitbx::af::shared<double> img_tof;
    vec2<std::size_t> image_size;
    int n_panels;

    TOFGeometryContext(Experiment& experiment, ImageSequence& data)
        : detector(*experiment.get_detector()) {
      Scan scan = *experiment.get_scan();

      std::shared_ptr<dxtbx::model::BeamBase> beam_ptr = experiment.get_beam();
      std::shared_ptr<PolychromaticBeam> beam =
        std::dynamic_pointer_cast<PolychromaticBeam>(beam_ptr);
      DIALS_ASSERT(beam != nullptr);
      unit_s0 = beam->get_unit_s0();
      sample_to_source_distance = beam->get_sample_to_source_distance();

      img_tof = scan.get_property<double>("time_of_flight");

      n_panels = detector.size();
      int num_images = data.size();
      image_size = detector[0].get_image_size();
      DIALS_ASSERT(num_images == img_tof.size());
    }
  };

  // First pass over a shoebox counts foreground/background pixels
  struct ShoeboxPixelCount {
    bool success = true;
    int n_signal = 0;
    int n_background = 0;
    scitbx::af::shared<double> incident_spectrum;
    scitbx::af::shared<double> empty_spectrum;
    scitbx::af::shared<std::size_t> n_contrib;
  };

  inline ShoeboxPixelCount scan_shoebox_pixels(
    const Shoebox<>& shoebox,
    const int6& bbox,
    const vec2<std::size_t>& image_size,
    const Shoebox<>* incident_shoebox = nullptr,
    const Shoebox<>* empty_shoebox = nullptr) {
    ShoeboxPixelCount result;
    bool has_incident = incident_shoebox != nullptr;
    if (has_incident) {
      result.incident_spectrum = scitbx::af::shared<double>(shoebox.zsize(), 0.0);
      result.empty_spectrum = scitbx::af::shared<double>(shoebox.zsize(), 0.0);
      result.n_contrib = scitbx::af::shared<std::size_t>(shoebox.zsize(), 0);
    }

    int bg_code = Valid | Background | BackgroundUsed;

    // Shoebox data are ordered (z, y, x)
    for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
      if (!result.success) {
        break;
      }
      for (std::size_t y = 0; y < shoebox.ysize(); ++y) {
        int panel_y = bbox[2] + y;
        if (panel_y > image_size[1] || panel_y < 0) {
          continue;
        }
        for (std::size_t x = 0; x < shoebox.xsize(); ++x) {
          int panel_x = bbox[0] + x;
          if (panel_x > image_size[0] || panel_x < 0) {
            continue;
          }

          if (has_incident) {
            result.incident_spectrum[z] += incident_shoebox->data(z, y, x);
            result.empty_spectrum[z] += empty_shoebox->data(z, y, x);
            result.n_contrib[z]++;
          }

          int mask = shoebox.mask(z, y, x);
          if ((mask & Foreground) == Foreground) {
            if ((mask & Valid) == Valid && (mask & Overlapped) == 0) {
              result.n_signal++;
            } else {
              result.success = false;
            }
          } else if ((mask & bg_code) == bg_code) {
            result.n_background++;
          }
        }
      }
    }
    return result;
  }

  // Loads imagesequence data into reflection shoeboxes
  inline dials::af::reflection_table load_shoebox_image_data(
    dials::af::reflection_table& reflection_table,
    std::shared_ptr<ImageSequence> reference_data,
    int n_panels,
    int num_images) {
    boost::python::dict d;
    dials::af::reflection_table ref_table =
      dxtbx::af::flex_table_suite::deepcopy(reflection_table, d);

    dials::af::ref<Shoebox<>> ref_shoeboxes = ref_table["shoebox"];
    for (std::size_t i = 0; i < ref_table.size(); ++i) {
      ref_shoeboxes[i].deallocate();
    }

    ShoeboxProcessor processor(ref_table, n_panels, 0, num_images, false);

    for (std::size_t img_num = 0; img_num < static_cast<std::size_t>(num_images);
         ++img_num) {
      dxtbx::format::Image<double> img = reference_data->get_corrected_data(img_num);
      dxtbx::format::Image<bool> mask = reference_data->get_mask(img_num);

      dials::af::shared<scitbx::af::versa<double, scitbx::af::c_grid<2>>> output_data(
        n_panels);
      dials::af::shared<scitbx::af::versa<bool, scitbx::af::c_grid<2>>> output_mask(
        n_panels);

      for (std::size_t i = 0; i < output_data.size(); ++i) {
        output_data[i] = img.tile(i).data();
        output_mask[i] = mask.tile(i).data();
      }
      processor.next_data_only(
        dials::model::Image<double>(output_data.const_ref(), output_mask.const_ref()));
    }

    return ref_table;
  }

  struct PixelCorrection {
    double I = 0;
    double B = 0;
    double var_I = 0;
    double var_B = 0;
    double I_raw = 0;
  };

  class PixelCorrector {
  public:
    PixelCorrector(const TOFGeometryContext& geometry,
                   bool apply_lorentz_correction,
                   double lorentz_wavelength_power,  // 4.0 = the full factor
                   int n_signal,
                   int n_background,
                   const dials_scaling::TOFIncidentSpectrumParams* incident_params,
                   const dials_scaling::TOFAbsorptionParams* absorption_params,
                   const scitbx::af::shared<double>* smoothed_incident,
                   const scitbx::af::shared<double>* smoothed_empty,
                   const scitbx::af::shared<std::size_t>* n_contrib)
        : geometry_(geometry),
          apply_lorentz_correction_(apply_lorentz_correction),
          lorentz_wavelength_power_(lorentz_wavelength_power),
          n_signal_(n_signal),
          n_background_(n_background),
          incident_params_(incident_params),
          absorption_params_(absorption_params),
          smoothed_incident_(smoothed_incident),
          smoothed_empty_(smoothed_empty),
          n_contrib_(n_contrib) {}

    // Returns false if the pixel cannot be corrected and should be skipped
    bool correct(int panel,
                 int panel_x,
                 int panel_y,
                 int frame_z,
                 std::size_t z,
                 double tof_us,
                 double raw_S,
                 double raw_B,
                 PixelCorrection& out) const {
      double tof_s = tof_us * std::pow(10, -6);

      double var_S = std::abs(raw_S);
      double var_B = std::abs(raw_B);
      if (n_background_ > 0) {
        var_B *= (1.0 + double(n_signal_) / double(n_background_));
      }

      bool has_incident = incident_params_ != nullptr;
      bool has_absorption = absorption_params_ != nullptr;

      // wl/two_theta are needed for the Lorentz correction and for the
      // absorption correction, so compute them once if either is active.
      double wl = 0.0, two_theta = 0.0, L = 1.0;
      if (apply_lorentz_correction_ || has_absorption) {
        scitbx::vec3<double> s1 = geometry_.detector[panel].get_pixel_lab_coord(
          scitbx::vec2<double>(panel_x, panel_y));
        double distance =
          (s1.length() + geometry_.sample_to_source_distance) * std::pow(10, -3);
        wl = ((Planck * tof_s) / (m_n * distance)) * std::pow(10, 10);
        two_theta = geometry_.detector[panel].get_two_theta_at_pixel(
          geometry_.unit_s0, scitbx::vec2<double>(panel_x, panel_y));
        if (apply_lorentz_correction_) {
          /*
           * sin^2(theta) / lambda^p.  The exponent is a parameter because the
           * two halves of the factor behave quite differently here: sin^2(theta)
           * barely varies across a shoebox, while lambda^-4 applied per ToF
           * slice reweights the inside of a peak and distorts the very profile
           * that is about to be fitted.  The wavelength part is also degenerate
           * with lawless's normalisation curve, which can absorb it.
           */
          double sin_two_theta_sq = std::pow(sin(two_theta * .5), 2);
          L = sin_two_theta_sq;
          if (lorentz_wavelength_power_ != 0.0) {
            L /= std::pow(wl, lorentz_wavelength_power_);
          }
        }
      }

      // Spherical absorption correction
      double T = 1.0;
      int two_theta_idx = 0;
      if (has_absorption) {
        double two_theta_deg = two_theta * (180.0 / pi);
        two_theta_idx = static_cast<int>(two_theta_deg / 10);
        double sample_muR =
          (absorption_params_->sample_linear_scattering_c
           + (absorption_params_->sample_linear_absorption_c / 1.8) * wl)
          * absorption_params_->sample_radius;
        T = dials_scaling::tof_pixel_spherical_absorption_correction(
          sample_muR, two_theta, two_theta_idx);
        if (T < 1e-7) {
          return false;
        }
      }

      double I0, var_I0, I_raw, B, var_B_pre;
      double J = 1.0, var_J = 0.0;

      // Bin-width normalisation is used when there is no incident/empty
      if (!has_incident) {
        double bin_width = geometry_.img_tof[frame_z] - geometry_.img_tof[frame_z - 1];
        I0 = (raw_S - raw_B) / bin_width;
        var_I0 = (var_S + var_B) / (bin_width * bin_width);
        I_raw = raw_S / bin_width;
        B = raw_B / bin_width;
        var_B_pre = var_B / (bin_width * bin_width);
      } else {
        double n = double((*n_contrib_)[z]);
        double raw_J = (*smoothed_incident_)[z] / n;
        double raw_E = (*smoothed_empty_)[z] / n;

        double sample_pc = incident_params_->sample_proton_charge;
        double incident_pc = incident_params_->incident_proton_charge;
        double empty_pc = incident_params_->empty_proton_charge;

        double S = raw_S / sample_pc;
        double Bn = raw_B / sample_pc;
        double Jn = raw_J / incident_pc;
        double En = raw_E / empty_pc;

        double var_Sn = var_S / (sample_pc * sample_pc);
        double var_Bn = var_B / (sample_pc * sample_pc);
        double var_Jn = (std::abs(raw_J) * n) / (incident_pc * incident_pc * n * n);
        double var_En = (std::abs(raw_E) * n) / (empty_pc * empty_pc * n * n);

        if (has_absorption) {
          // Spherical absorption correction for the incident spectrum
          double incident_muR =
            (absorption_params_->incident_linear_scattering_c
             + (absorption_params_->incident_linear_absorption_c / 1.8) * wl)
            * absorption_params_->incident_radius;
          double J_T = dials_scaling::tof_pixel_spherical_absorption_correction(
            incident_muR, two_theta, two_theta_idx);
          if (J_T < 1e-7) {
            return false;
          }
          Jn /= J_T;
          if (Jn < 1e-7) {
            return false;
          }
          var_Jn /= (J_T * J_T);
        }

        I0 = S - Bn - En;
        var_I0 = var_Sn + var_Bn + var_En;
        I_raw = S;
        B = Bn;
        var_B_pre = var_Bn;
        J = Jn;
        var_J = var_Jn;
      }

      // Apply the Lorentz correction and normalise
      double JT = J * T;
      out.I = L * I0 / JT;
      out.B = L * B / JT;
      out.I_raw = L * I_raw / JT;
      out.var_I =
        (L * L / (JT * JT)) * var_I0 + (L * L * I0 * I0 / std::pow(JT, 4)) * var_J;
      out.var_B =
        (L * L / (JT * JT)) * var_B_pre + (L * L * B * B / std::pow(JT, 4)) * var_J;
      return true;
    }

  private:
    const TOFGeometryContext& geometry_;
    bool apply_lorentz_correction_;
    double lorentz_wavelength_power_;
    int n_signal_;
    int n_background_;
    const dials_scaling::TOFIncidentSpectrumParams* incident_params_;
    const dials_scaling::TOFAbsorptionParams* absorption_params_;
    const scitbx::af::shared<double>* smoothed_incident_;
    const scitbx::af::shared<double>* smoothed_empty_;
    const scitbx::af::shared<std::size_t>* n_contrib_;
  };

  struct ShoeboxIntegrationResult {
    bool success = true;
    double intensity = 0.0;
    double variance = 0.0;
    scitbx::af::shared<double> tof_z;
    scitbx::af::shared<double> projected_intensity;
    scitbx::af::shared<double> raw_projected_intensity;
    scitbx::af::shared<double> projected_background;
    scitbx::af::versa<double, af::c_grid<3>> intensity_3d;
    scitbx::af::versa<double, af::c_grid<3>> background_var_3d;
    scitbx::af::versa<vec3<double>, af::c_grid<3>> coords_3d;
  };

  // Background variance given to a voxel that belongs to a neighbouring
  // reflection. The 3D fitters weight by 1 / (|obs| + var_b), so this removes
  // the voxel from the fit while staying finite.
  const double overlapped_voxel_variance = 1e30;

  // Second pass over a shoebox
  // applies the pixel corrector and accumulates summation intensity/variance
  // plus every array needed by 1D/3D profile fitting
  inline ShoeboxIntegrationResult integrate_shoebox(const Shoebox<>& shoebox,
                                                    const int6& bbox,
                                                    const TOFGeometryContext& geometry,
                                                    const PixelCorrector& corrector,
                                                    bool initial_success = true) {
    ShoeboxIntegrationResult result;
    result.success = initial_success;
    result.tof_z = scitbx::af::shared<double>(shoebox.zsize(), 0.0);
    result.projected_intensity = scitbx::af::shared<double>(shoebox.zsize(), 0.0);
    result.raw_projected_intensity = scitbx::af::shared<double>(shoebox.zsize(), 0.0);
    result.projected_background = scitbx::af::shared<double>(shoebox.zsize(), 0.0);

    auto acc = shoebox.data.accessor();
    af::c_grid<3> transposed_acc(acc[2], acc[1], acc[0]);
    result.intensity_3d = scitbx::af::versa<double, af::c_grid<3>>(transposed_acc);
    result.background_var_3d = scitbx::af::versa<double, af::c_grid<3>>(transposed_acc);
    result.coords_3d = scitbx::af::versa<vec3<double>, af::c_grid<3>>(transposed_acc);

    int panel = shoebox.panel;

    // Shoebox data are ordered (z, y, x)
    for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
      double intensity_z = 0;
      double intensity_raw_z = 0;
      double background_z = 0;

      if (!result.success) {
        break;
      }

      int frame_z = bbox[4] + z;
      double tof = geometry.img_tof[frame_z];
      result.tof_z[z] = tof;  // (us)

      for (std::size_t y = 0; y < shoebox.ysize(); ++y) {
        int panel_y = bbox[2] + y;
        if (panel_y > geometry.image_size[1] || panel_y < 0) {
          continue;
        }
        for (std::size_t x = 0; x < shoebox.xsize(); ++x) {
          int panel_x = bbox[0] + x;
          if (panel_x > geometry.image_size[0] || panel_x < 0) {
            continue;
          }

          double raw_S = shoebox.data(z, y, x);
          double raw_B = shoebox.background(z, y, x);
          int mask = shoebox.mask(z, y, x);

          PixelCorrection pixel;
          if (!corrector.correct(
                panel, panel_x, panel_y, frame_z, z, tof, raw_S, raw_B, pixel)) {
            continue;
          }

          /*
           * A voxel flagged Overlapped outside the foreground belongs to a
           * neighbouring reflection or holds its peak (see tof_overlaps.h).
           * It is kept out of the projections a 1D profile is fitted to, and
           * given no value and no weight in the arrays a 3D profile is fitted
           * to. Inside the foreground it fails the reflection below, as before.
           */
          bool overlapped = (mask & Overlapped) != 0;
          if (!overlapped) {
            intensity_z += pixel.I;
            intensity_raw_z += pixel.I_raw;
            background_z += pixel.B;
          }

          result.intensity_3d(x, y, z) = overlapped ? 0.0 : pixel.I;
          result.background_var_3d(x, y, z) =
            overlapped ? overlapped_voxel_variance : pixel.var_B;
          double x_c = x + shoebox.xoffset() + 0.5;
          double y_c = y + shoebox.yoffset() + 0.5;
          double z_c = z + shoebox.zoffset() + 0.5;
          result.coords_3d(x, y, z) = vec3<double>(x_c, y_c, z_c);

          // Accumulate summation values if pixel in foreground and valid
          if ((mask & Foreground) == Foreground && (mask & Valid) == Valid
              && (mask & Overlapped) == 0) {
            result.intensity += pixel.I;
            result.variance += pixel.var_I;
          } else if ((mask & Foreground) == Foreground) {
            result.success = false;
            break;
          }
        }
      }

      result.projected_intensity[z] = intensity_z;
      result.raw_projected_intensity[z] = intensity_raw_z;
      result.projected_background[z] = background_z;
    }

    return result;
  }

  /*
   * One entry of a profile library: a peak shape and where on the detector it
   * was measured.  Shapes vary across the detector, so a weak reflection takes
   * the shape of the nearest strong one, as Mantid keys its library on
   * (phi, theta).
   */
  struct ProfileLibraryEntry {
    int panel = 0;
    double x = 0.0;
    double y = 0.0;
    IBIXShape shape;

    /*
     * The shape of the peak on the detector face, as a bivariate Gaussian, and
     * how far the peak sat from where it was predicted. Only needed to model a
     * reflection in 3D (see tof_overlap_deconvolution.h), and only set, with
     * has_xy, where the measurement converged.
     */
    bool has_xy = false;
    double sig_x = 0.0;
    double sig_y = 0.0;
    double sig_p = 0.0;
    double dx = 0.0;  // observed minus predicted position (px)
    double dy = 0.0;
    double dt = 0.0;  // centroid of the fitted line shape minus predicted ToF (us)
  };

  /*
   * The centroid of the ToF line shape relative to its T_ph parameter. The
   * shape is a rising exponential (rate alpha) joined to a falling one (rate
   * beta), convolved with a Gaussian, whose mean is 1/beta - 1/alpha. T_ph is
   * therefore not where the peak is, and how far off it is depends on the
   * shape, so the position of a peak is carried as its centroid: that can be
   * compared between reflections of different shape.
   */
  inline double ibix_centroid_offset(const IBIXShape& shape) {
    return 1.0 / shape.beta - 1.0 / shape.alpha;
  }

  /*
   * What is expected of a reflection that cannot be measured on its own: its
   * shape in ToF and on the detector face, and where its peak sits relative
   * to its prediction. See Profile1DProfilelibFitter::local_peak_model.
   */
  struct LocalPeakModel {
    IBIXShape shape;
    double sig_x = 0.0;
    double sig_y = 0.0;
    double sig_p = 0.0;
    double dx = 0.0;  // expected peak minus predicted position (px)
    double dy = 0.0;
    double dt = 0.0;  // expected centroid minus predicted ToF (us)
  };

  // Where a reflection is predicted to be, and where its shoebox starts
  struct ReflectionGeometry {
    double x_cal = 0.0;  // px
    double y_cal = 0.0;
    double tof_cal = 0.0;  // us
    int x0 = 0;            // first pixel of the bounding box
    int y0 = 0;
  };

  /*
   * The shape of a peak on the detector face, as a bivariate Gaussian.
   *
   * Measured on the background-subtracted shoebox summed along ToF over the
   * slices where the fitted line profile is at least a tenth of its maximum,
   * by adaptive moments: the moments are taken under a Gaussian weight, the
   * weight is set to the Gaussian those moments imply, and the two are
   * iterated to agreement. Plain second moments over a padded box are
   * dominated by the noise far from the peak; a weight matched to the peak
   * removes that without needing a threshold. For a Gaussian peak of
   * covariance T under a weight of covariance W the weighted moments are
   * (T^-1 + W^-1)^-1, which is inverted for T at each step.
   *
   * Returns false if there is no positive signal, or the iteration leaves the
   * positive-definite matrices or does not converge.
   */
  inline bool measure_xy_shape(const ShoeboxIntegrationResult& shoebox_result,
                               const scitbx::af::shared<double>& line_profile,
                               int x0,
                               int y0,
                               double& cx,
                               double& cy,
                               double& sig_x,
                               double& sig_y,
                               double& sig_p) {
    const std::size_t nx = shoebox_result.intensity_3d.accessor()[0];
    const std::size_t ny = shoebox_result.intensity_3d.accessor()[1];
    const std::size_t nz = shoebox_result.intensity_3d.accessor()[2];
    if (nx < 3 || ny < 3 || line_profile.size() != nz) {
      return false;
    }
    double peak = 0.0;
    for (std::size_t iz = 0; iz < nz; ++iz) {
      if (std::isfinite(line_profile[iz])) {
        peak = std::max(peak, line_profile[iz]);
      }
    }
    if (!(peak > 0.0)) {
      return false;
    }

    std::vector<double> image(nx * ny, 0.0);
    for (std::size_t iz = 0; iz < nz; ++iz) {
      if (!(line_profile[iz] >= 0.1 * peak)) {
        continue;
      }
      for (std::size_t ix = 0; ix < nx; ++ix) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
          image[ix * ny + iy] += shoebox_result.intensity_3d(ix, iy, iz);
        }
      }
    }

    // Starting point: plain moments of the positive part
    double s0 = 0.0, sx = 0.0, sy = 0.0;
    for (std::size_t ix = 0; ix < nx; ++ix) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        double v = image[ix * ny + iy];
        if (v > 0.0) {
          s0 += v;
          sx += v * (x0 + ix + 0.5);
          sy += v * (y0 + iy + 0.5);
        }
      }
    }
    if (!(s0 > 0.0)) {
      return false;
    }
    cx = sx / s0;
    cy = sy / s0;
    double wxx = 0.0, wyy = 0.0, wxy = 0.0;
    for (std::size_t ix = 0; ix < nx; ++ix) {
      for (std::size_t iy = 0; iy < ny; ++iy) {
        double v = image[ix * ny + iy];
        if (v > 0.0) {
          double dx = x0 + ix + 0.5 - cx;
          double dy = y0 + iy + 0.5 - cy;
          wxx += v * dx * dx;
          wyy += v * dy * dy;
          wxy += v * dx * dy;
        }
      }
    }
    wxx = std::max(wxx / s0, 0.25);
    wyy = std::max(wyy / s0, 0.25);
    wxy = wxy / s0;
    if (wxy * wxy >= 0.9 * wxx * wyy) {
      wxy = 0.0;
    }

    bool converged = false;
    for (int iteration = 0; iteration < 100 && !converged; ++iteration) {
      double det_w = wxx * wyy - wxy * wxy;
      if (!(det_w > 0.0)) {
        return false;
      }
      double ixx = wyy / det_w, iyy = wxx / det_w, ixy = -wxy / det_w;

      double m0 = 0.0, mx = 0.0, my = 0.0, mxx = 0.0, myy = 0.0, mxy = 0.0;
      for (std::size_t ix = 0; ix < nx; ++ix) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
          double dx = x0 + ix + 0.5 - cx;
          double dy = y0 + iy + 0.5 - cy;
          double w =
            std::exp(-0.5 * (ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy));
          double wv = w * image[ix * ny + iy];
          m0 += wv;
          mx += wv * dx;
          my += wv * dy;
          mxx += wv * dx * dx;
          myy += wv * dy * dy;
          mxy += wv * dx * dy;
        }
      }
      if (!(m0 > 0.0)) {
        return false;
      }
      mx /= m0;
      my /= m0;
      mxx = mxx / m0 - mx * mx;
      myy = myy / m0 - my * my;
      mxy = mxy / m0 - mx * my;
      double det_m = mxx * myy - mxy * mxy;
      if (!(det_m > 0.0) || !(mxx > 0.0)) {
        return false;
      }

      // T^-1 = M^-1 - W^-1
      double txx_inv = myy / det_m - ixx;
      double tyy_inv = mxx / det_m - iyy;
      double txy_inv = -mxy / det_m - ixy;
      double det_t_inv = txx_inv * tyy_inv - txy_inv * txy_inv;
      if (!(det_t_inv > 0.0) || !(txx_inv > 0.0)) {
        return false;
      }
      double txx = tyy_inv / det_t_inv;
      double tyy = txx_inv / det_t_inv;
      double txy = -txy_inv / det_t_inv;

      // Under a weight matched to the peak the weighted centroid moves half
      // of the way to the true one
      double shift_x = 2.0 * mx;
      double shift_y = 2.0 * my;
      double change = std::abs(txx - wxx) / wxx + std::abs(tyy - wyy) / wyy
                      + std::abs(shift_x) + std::abs(shift_y);
      converged = change < 1e-4;

      cx += shift_x;
      cy += shift_y;
      wxx = txx;
      wyy = tyy;
      wxy = txy;
      if (!std::isfinite(change) || cx < x0 || cx > x0 + double(nx) || cy < y0
          || cy > y0 + double(ny)) {
        return false;
      }
    }
    if (!converged) {
      return false;
    }

    sig_x = std::sqrt(wxx);
    sig_y = std::sqrt(wyy);
    sig_p = wxy / (sig_x * sig_y);
    return std::isfinite(sig_x) && std::isfinite(sig_y) && std::isfinite(sig_p)
           && sig_x >= 0.2 && sig_y >= 0.2 && sig_x <= double(nx) && sig_y <= double(ny)
           && std::abs(sig_p) < 0.95;
  }

  /*
   * Variance of a profile-fitted intensity.
   *
   * With the shape held fixed the intensity is linear in the amplitude, so
   * weighted least squares gives its error exactly: for data with per-slice
   * variances v_i and a model m_i, var(I) = I^2 / sum(m_i^2 / v_i).
   *
   * The per-slice variances are not carried through the corrections, so they are
   * taken as the summation variance distributed across the slices in proportion
   * to counts plus twice background. That keeps the absolute scale tied to the
   * summation variance, which is the one checked against the scatter of symmetry
   * equivalents, while letting the profile do what it should: a slice the fitted
   * peak says holds no signal stops contributing noise. The result is at most the
   * summation variance and below it when the peak is concentrated, which is where
   * copying the summation variance was overestimating sigma.
   */
  inline double profile_fit_variance(const ShoeboxIntegrationResult& shoebox_result,
                                     const scitbx::af::shared<double>& model,
                                     double I_prf) {
    const std::size_t n = model.size();
    if (n == 0 || !(shoebox_result.variance > 0.0) || !(I_prf > 0.0)) {
      return 0.0;
    }
    if (shoebox_result.projected_intensity.size() != n
        || shoebox_result.projected_background.size() != n) {
      return 0.0;
    }

    double weight_total = 0.0;
    scitbx::af::shared<double> weight(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      double signal = std::max(shoebox_result.projected_intensity[i], 0.0);
      double background = std::max(shoebox_result.projected_background[i], 0.0);
      weight[i] = signal + 2.0 * background;
      weight_total += weight[i];
    }
    if (!(weight_total > 0.0)) {
      return 0.0;
    }

    double inv_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double v = shoebox_result.variance * weight[i] / weight_total;
      if (!(v > 0.0) || !std::isfinite(model[i])) {
        continue;
      }
      inv_sum += (model[i] * model[i]) / v;
    }
    if (!(inv_sum > 0.0)) {
      return 0.0;
    }
    double var = (I_prf * I_prf) / inv_sum;
    if (!std::isfinite(var) || var <= 0.0) {
      return 0.0;
    }
    // Never claim to beat counting statistics over the whole box
    return std::min(var, shoebox_result.variance);
  }

  // Profile fitting strategy
  class ProfileFitter {
  public:
    virtual ~ProfileFitter() {}
    virtual bool fit(const ShoeboxIntegrationResult& shoebox_result,
                     double& I_prf,
                     double& var_prf) = 0;

    /*
     * Fitters that build a profile library need the table twice: once to
     * measure shapes on strong reflections, and once to apply them. Fitters
     * that do not simply leave these alone.
     */
    virtual bool needs_two_passes() const {
      return false;
    }

    // Pass one: offer a reflection as a candidate library entry
    // geometry, where known, lets the entry also record the shape of the peak
    // on the detector face and its offset from the prediction
    virtual void collect(const ShoeboxIntegrationResult& shoebox_result,
                         int panel,
                         double x,
                         double y,
                         const ReflectionGeometry* geometry = nullptr) {}

    // The library entry measured nearest to (x, y), on the same panel where
    // there is one. With require_xy, only entries that also hold the shape on
    // the detector face are considered.
    virtual const ProfileLibraryEntry* nearest_entry(int panel,
                                                     double x,
                                                     double y,
                                                     bool require_xy = false) const {
      return nullptr;
    }

    // Pass two: fit with the shape taken from the library
    virtual bool fit_with_library(const ShoeboxIntegrationResult& shoebox_result,
                                  int panel,
                                  double x,
                                  double y,
                                  double& I_prf,
                                  double& var_prf) {
      return fit(shoebox_result, I_prf, var_prf);
    }

    virtual std::size_t library_size() const {
      return 0;
    }

    // How many library entries also hold the shape on the detector face
    virtual std::size_t library_size_xy() const {
      return 0;
    }

    // Called once, when pass one is complete and the library will not change
    virtual void finalise_library() {}

    // The shape and position to expect of a reflection at (x, y) on a panel.
    // False if the library cannot say.
    virtual bool local_peak_model(int panel,
                                  double x,
                                  double y,
                                  LocalPeakModel& model) const {
      return false;
    }
  };

  class Profile1DIBIXFitter : public ProfileFitter {
  public:
    explicit Profile1DIBIXFitter(TOFProfile1DIBIXParams params) : params_(params) {}

    bool fit(const ShoeboxIntegrationResult& shoebox_result,
             double& I_prf,
             double& var_prf) override {
      var_prf = shoebox_result.variance;  // Use summation variance as approximation
      return fit_profile_1d_ibix(shoebox_result.projected_intensity.const_ref(),
                                 shoebox_result.tof_z.const_ref(),
                                 params_,
                                 I_prf);
    }

  private:
    TOFProfile1DIBIXParams params_;
  };

  /*
   * Profile fitting against a library of shapes measured on strong reflections,
   * after Mantid's IntegratePeaksProfileFitting: fit the peak shape where there
   * are enough counts to determine it, store it against where it was measured,
   * then hold it fixed and fit only the amplitude and position everywhere else.
   *
   * The peak shape is the same back-to-back exponential profile_1d_ibix fits,
   * so the two share their model and their fitting machinery; what differs is
   * that a weak reflection is never asked to determine its own shape.
   *
   * library_min_corr is the correlation a free fit must reach for its shape to
   * enter the library.  It is deliberately looser than the threshold used to
   * accept an intensity: a shape only has to be the right shape, and 0.9 is met
   * by almost any smooth curve over the seven points of a coarse projection
   * while being close to unreachable over fifty fine ones.
   */
  class Profile1DProfilelibFitter : public ProfileFitter {
  public:
    explicit Profile1DProfilelibFitter(TOFProfile1DProfilelibParams params)
        : params_(params),
          min_i_sigma_(params.library_min_i_sigma),
          min_corr_(params.library_min_corr) {}

    bool needs_two_passes() const override {
      return true;
    }

    // Pass one: fit freely, and keep the shape if the reflection is strong
    // enough to have determined it
    void collect(const ShoeboxIntegrationResult& shoebox_result,
                 int panel,
                 double x,
                 double y,
                 const ReflectionGeometry* geometry = nullptr) override {
      if (!shoebox_result.success || !(shoebox_result.variance > 0.0)) {
        return;
      }
      double i_sigma = shoebox_result.intensity / std::sqrt(shoebox_result.variance);
      if (!(i_sigma >= min_i_sigma_)) {
        return;
      }

      TOFProfile1DProfilelibParams local = params_;  // the fit mutates nothing shared
      local.trust_min_corr = min_corr_;              // see Profile1DProfilelibFitter
      double I_prf = 0.0;
      IBIXShape shape;
      scitbx::af::shared<double> line_profile(shoebox_result.tof_z.size(), 0.0);
      bool ok =
        fit_profile_1d_profilelib_shape(shoebox_result.projected_intensity.const_ref(),
                                        shoebox_result.tof_z.const_ref(),
                                        local,
                                        I_prf,
                                        line_profile,
                                        false,
                                        &shape);
      if (!ok || !shape.valid) {
        return;
      }

      ProfileLibraryEntry entry;
      entry.panel = panel;
      entry.x = x;
      entry.y = y;
      entry.shape = shape;

      if (geometry != nullptr) {
        double cx = 0.0, cy = 0.0;
        if (measure_xy_shape(shoebox_result,
                             line_profile,
                             geometry->x0,
                             geometry->y0,
                             cx,
                             cy,
                             entry.sig_x,
                             entry.sig_y,
                             entry.sig_p)) {
          entry.has_xy = true;
          entry.dx = cx - geometry->x_cal;
          entry.dy = cy - geometry->y_cal;
          entry.dt = shape.T_ph + ibix_centroid_offset(shape) - geometry->tof_cal;
        }
      }

      std::lock_guard<std::mutex> lock(mutex_);
      library_.push_back(entry);
    }

    // Pass two: take the shape of the nearest library entry, on the same panel
    // where possible, and fit only the amplitude and position
    bool fit_with_library(const ShoeboxIntegrationResult& shoebox_result,
                          int panel,
                          double x,
                          double y,
                          double& I_prf,
                          double& var_prf) override {
      var_prf = shoebox_result.variance;
      const ProfileLibraryEntry* entry = nearest_entry(panel, x, y);
      if (entry == nullptr) {
        return false;
      }
      std::size_t n = shoebox_result.projected_intensity.size();
      scitbx::af::shared<double> model(n, 0.0);
      bool ok =
        fit_profile_1d_profilelib_forced(shoebox_result.projected_intensity.const_ref(),
                                         shoebox_result.tof_z.const_ref(),
                                         entry->shape,
                                         params_.A_min,
                                         params_.A_max,
                                         I_prf,
                                         model);
      if (!ok) {
        return false;
      }
      double fitted_var = profile_fit_variance(shoebox_result, model, I_prf);
      if (fitted_var > 0.0) {
        var_prf = fitted_var;
      }
      return true;
    }

    // Only reached if the library is empty
    bool fit(const ShoeboxIntegrationResult& shoebox_result,
             double& I_prf,
             double& var_prf) override {
      var_prf = shoebox_result.variance;
      TOFProfile1DProfilelibParams local = params_;
      return fit_profile_1d_profilelib_shape(
        shoebox_result.projected_intensity.const_ref(),
        shoebox_result.tof_z.const_ref(),
        local,
        I_prf);
    }

    std::size_t library_size() const override {
      return library_.size();
    }

    std::size_t library_size_xy() const override {
      std::size_t n = 0;
      for (const auto& e : library_) {
        if (e.has_xy) {
          ++n;
        }
      }
      return n;
    }

    /*
     * Summarises the detector-face shapes and the position offsets of the
     * library, per panel where a panel holds enough entries and over the whole
     * detector otherwise, and measures how far individual reflections scatter
     * about those offsets.
     *
     * The offset of one library reflection from its prediction is mostly that
     * reflection's own prediction error, not something its neighbours share:
     * on MANDI the scatter between reflections is 0.7 px and 0.7 ToF frames
     * while panel medians differ by 0.1 to 0.2. So what a neighbour can be told
     * is the typical offset and how uncertain its own position is, and both
     * are taken from robust statistics rather than from any single entry.
     */
    void finalise_library() override {
      std::lock_guard<std::mutex> lock(mutex_);
      panel_statistics_.clear();
      have_statistics_ = false;

      std::map<int, std::vector<const ProfileLibraryEntry*>> by_panel;
      std::vector<const ProfileLibraryEntry*> all;
      for (const auto& e : library_) {
        if (e.has_xy) {
          by_panel[e.panel].push_back(&e);
          all.push_back(&e);
        }
      }
      if (all.empty()) {
        return;
      }
      global_statistics_ = summarise(all);
      for (const auto& item : by_panel) {
        if (item.second.size() >= min_entries_per_panel_) {
          panel_statistics_[item.first] = summarise(item.second);
        }
      }

      // Scatter of each entry about the offset its panel would predict for it
      std::vector<double> rx, ry, rt;
      for (const ProfileLibraryEntry* e : all) {
        const LibraryStatistics& stats = statistics_for(e->panel);
        rx.push_back(e->dx - stats.dx);
        ry.push_back(e->dy - stats.dy);
        rt.push_back(e->dt - stats.dt);
      }
      scatter_x_ = robust_sd(rx);
      scatter_y_ = robust_sd(ry);
      scatter_t_ = robust_sd(rt);
      have_statistics_ = true;
    }

    /*
     * The ToF shape of the nearest library entry, the detector-face shape and
     * the offsets typical of the panel, with each width broadened by the
     * scatter of individual positions about those offsets.
     *
     * The broadening is what makes a fit at a fixed position usable. A
     * reflection sits at its prediction plus an error nobody knows; fitting a
     * sharp profile at the wrong place loses intensity, every time, so the
     * result is biased low. Averaged over the distribution of that error, the
     * peak is its own shape convolved with the distribution, and a linear fit
     * of that broadened profile recovers the intensity without bias. For
     * Gaussian errors the convolution only adds variances, in ToF as well,
     * where the line shape already is a function convolved with a Gaussian.
     */
    bool local_peak_model(int panel,
                          double x,
                          double y,
                          LocalPeakModel& model) const override {
      if (!have_statistics_) {
        return false;
      }
      const ProfileLibraryEntry* entry = nearest_entry(panel, x, y, true);
      if (entry == nullptr || !entry->shape.valid) {
        return false;
      }
      const LibraryStatistics& stats = statistics_for(panel);
      model.shape = entry->shape;
      model.shape.sigma =
        std::sqrt(entry->shape.sigma * entry->shape.sigma + scatter_t_ * scatter_t_);
      model.sig_x = std::sqrt(stats.sig_x * stats.sig_x + scatter_x_ * scatter_x_);
      model.sig_y = std::sqrt(stats.sig_y * stats.sig_y + scatter_y_ * scatter_y_);
      model.sig_p =
        stats.sig_p * stats.sig_x * stats.sig_y / (model.sig_x * model.sig_y);
      model.dx = stats.dx;
      model.dy = stats.dy;
      model.dt = stats.dt;
      return std::isfinite(model.shape.sigma) && model.sig_x > 0.0 && model.sig_y > 0.0;
    }

    const ProfileLibraryEntry* nearest_entry(int panel,
                                             double x,
                                             double y,
                                             bool require_xy = false) const override {
      const ProfileLibraryEntry* best = nullptr;
      double best_d2 = std::numeric_limits<double>::infinity();
      // Prefer the same panel; fall back to any panel if it holds no shapes
      for (int same_panel = 1; same_panel >= 0; --same_panel) {
        for (const auto& e : library_) {
          if (same_panel && e.panel != panel) {
            continue;
          }
          if (require_xy && !e.has_xy) {
            continue;
          }
          double dx = e.x - x;
          double dy = e.y - y;
          double d2 = dx * dx + dy * dy;
          if (d2 < best_d2) {
            best_d2 = d2;
            best = &e;
          }
        }
        if (best != nullptr) {
          break;
        }
      }
      return best;
    }

  private:
    struct LibraryStatistics {
      double sig_x = 0.0, sig_y = 0.0, sig_p = 0.0;
      double dx = 0.0, dy = 0.0, dt = 0.0;
    };

    static double median(std::vector<double> values) {
      DIALS_ASSERT(!values.empty());
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
    static double robust_sd(const std::vector<double>& values) {
      double centre = median(values);
      std::vector<double> deviations(values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
        deviations[i] = std::abs(values[i] - centre);
      }
      return 1.4826 * median(deviations);
    }

    static LibraryStatistics summarise(
      const std::vector<const ProfileLibraryEntry*>& entries) {
      std::vector<double> sig_x, sig_y, sig_p, dx, dy, dt;
      for (const ProfileLibraryEntry* e : entries) {
        sig_x.push_back(e->sig_x);
        sig_y.push_back(e->sig_y);
        sig_p.push_back(e->sig_p);
        dx.push_back(e->dx);
        dy.push_back(e->dy);
        dt.push_back(e->dt);
      }
      LibraryStatistics stats;
      stats.sig_x = median(sig_x);
      stats.sig_y = median(sig_y);
      stats.sig_p = median(sig_p);
      stats.dx = median(dx);
      stats.dy = median(dy);
      stats.dt = median(dt);
      return stats;
    }

    const LibraryStatistics& statistics_for(int panel) const {
      std::map<int, LibraryStatistics>::const_iterator it =
        panel_statistics_.find(panel);
      return it != panel_statistics_.end() ? it->second : global_statistics_;
    }

    TOFProfile1DProfilelibParams params_;
    double min_i_sigma_;
    double min_corr_;
    std::vector<ProfileLibraryEntry> library_;
    mutable std::mutex mutex_;

    // Set by finalise_library
    const std::size_t min_entries_per_panel_ = 10;
    bool have_statistics_ = false;
    LibraryStatistics global_statistics_;
    std::map<int, LibraryStatistics> panel_statistics_;
    double scatter_x_ = 0.0;
    double scatter_y_ = 0.0;
    double scatter_t_ = 0.0;
  };

  class Profile1DICFitter : public ProfileFitter {
  public:
    explicit Profile1DICFitter(TOFProfile1DICParams params) : params_(params) {}

    bool fit(const ShoeboxIntegrationResult& shoebox_result,
             double& I_prf,
             double& var_prf) override {
      var_prf = shoebox_result.variance;  // Use summation variance as approximation
      return fit_profile_1d_ic(shoebox_result.projected_intensity.const_ref(),
                               shoebox_result.tof_z.const_ref(),
                               params_,
                               I_prf);
    }

  private:
    TOFProfile1DICParams params_;
  };

  class Profile3DGutmannFitter : public ProfileFitter {
  public:
    explicit Profile3DGutmannFitter(TOFProfile3DGutmannParams params)
        : params_(params) {}

    bool fit(const ShoeboxIntegrationResult& shoebox_result,
             double& I_prf,
             double& var_prf) override {
      var_prf = shoebox_result.variance;  // Use summation variance as approximation
      return fit_profile_3d_gutmann(shoebox_result.coords_3d.const_ref(),
                                    shoebox_result.intensity_3d,
                                    shoebox_result.background_var_3d,
                                    params_,
                                    I_prf);
    }

  private:
    TOFProfile3DGutmannParams params_;
  };

  class Profile3DICFitter : public ProfileFitter {
  public:
    explicit Profile3DICFitter(TOFProfile3DICParams params) : params_(params) {}

    bool fit(const ShoeboxIntegrationResult& shoebox_result,
             double& I_prf,
             double& var_prf) override {
      var_prf = shoebox_result.variance;  // Use summation variance as approximation
      return fit_profile_3d_ic(shoebox_result.coords_3d.const_ref(),
                               shoebox_result.intensity_3d,
                               shoebox_result.background_var_3d,
                               params_,
                               I_prf);
    }

  private:
    TOFProfile3DICParams params_;
  };

  class Profile3DIBIXFitter : public ProfileFitter {
  public:
    explicit Profile3DIBIXFitter(TOFProfile3DIBIXParams params) : params_(params) {}

    bool fit(const ShoeboxIntegrationResult& shoebox_result,
             double& I_prf,
             double& var_prf) override {
      var_prf = shoebox_result.variance;  // Use summation variance as approximation
      return fit_profile_3d_ibix(shoebox_result.coords_3d.const_ref(),
                                 shoebox_result.intensity_3d,
                                 shoebox_result.background_var_3d,
                                 params_,
                                 I_prf);
    }

  private:
    TOFProfile3DIBIXParams params_;
  };

  // Runs incident/empty scanning/smoothing for one shoebox and builds
  // the PixelCorrector
  struct ShoeboxCorrectorInputs {
    ShoeboxPixelCount shoebox_pixel_count;
    scitbx::af::shared<double> smoothed_incident;
    scitbx::af::shared<double> smoothed_empty;
  };

  inline ShoeboxCorrectorInputs prepare_shoebox_corrector_inputs(
    const Shoebox<>& shoebox,
    const int6& bbox,
    const vec2<std::size_t>& image_size,
    const Shoebox<>* incident_shoebox,
    const Shoebox<>* empty_shoebox) {
    ShoeboxCorrectorInputs inputs;
    inputs.shoebox_pixel_count =
      scan_shoebox_pixels(shoebox, bbox, image_size, incident_shoebox, empty_shoebox);
    if (incident_shoebox != nullptr) {
      // Smooth incident and empty to avoid dividing by a noisy signal
      inputs.smoothed_empty =
        dials_scaling::savitzky_golay(inputs.shoebox_pixel_count.empty_spectrum, 7, 2);
      inputs.smoothed_incident = dials_scaling::savitzky_golay(
        inputs.shoebox_pixel_count.incident_spectrum, 7, 2);
    }
    return inputs;
  }

  // Updates reflection_table with intensities and variances corrected by
  // any combination of an incident/empty spectrum normalisation, a
  // spherical absorption correction, and a Lorentz correction.
  // profile-fitted intensity is also produced if profile_fitter is
  // given
  inline void integrate_reflection_table(
    dials::af::reflection_table& reflection_table,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    const bool& apply_lorentz_correction,
    int n_threads,
    std::shared_ptr<ProfileFitter> profile_fitter = nullptr,
    double lorentz_wavelength_power = 4.0) {
    std::size_t n_reflections = reflection_table.size();
    TOFGeometryContext geometry(experiment, data);

    // incident and empty
    dials::af::reflection_table i_reflection_table;
    dials::af::reflection_table e_reflection_table;
    dials::af::shared<Shoebox<>> i_shoeboxes;
    dials::af::shared<Shoebox<>> e_shoeboxes;

    if (incident_params) {
      i_reflection_table = load_shoebox_image_data(reflection_table,
                                                   incident_params->incident_data,
                                                   geometry.n_panels,
                                                   data.size());
      e_reflection_table = load_shoebox_image_data(
        reflection_table, incident_params->empty_data, geometry.n_panels, data.size());
      i_shoeboxes = i_reflection_table["shoebox"];
      e_shoeboxes = e_reflection_table["shoebox"];
    }

    dials::af::shared<Shoebox<>> shoeboxes = reflection_table["shoebox"];
    dials::af::shared<std::size_t> refl_flags =
      reflection_table.get<std::size_t>("flags");
    dials::af::const_ref<int6> bboxes = reflection_table["bbox"];

    // Arrays to store output of integration
    dials::af::shared<bool> succeeded(reflection_table.size());
    dials::af::shared<double> intensities(reflection_table.size());
    dials::af::shared<double> variances(reflection_table.size());
    dials::af::shared<bool> succeeded_prf;
    dials::af::shared<double> intensities_prf;
    dials::af::shared<double> variances_prf;

    if (profile_fitter) {
      succeeded_prf.resize(reflection_table.size());
      intensities_prf.resize(reflection_table.size());
      variances_prf.resize(reflection_table.size());
    }

    // Progress counters, so that a long fit reports something while it runs
    std::atomic<std::size_t> n_done(0);
    std::atomic<std::size_t> n_prf_ok(0);
    std::atomic<std::size_t> n_collected(0);

    // Predicted positions, where the table has them: a library entry then also
    // records the shape of its peak on the detector face
    bool have_predictions =
      reflection_table.contains("xyzcal.px") && reflection_table.contains("tof_cal");
    dials::af::shared<vec3<double>> xyzcal_px;
    dials::af::shared<double> tof_cal;
    if (have_predictions) {
      xyzcal_px = reflection_table["xyzcal.px"];
      tof_cal = reflection_table["tof_cal"];
    }

    // Pass one, when the fitter builds a library: measure shapes on the strong
    // reflections before anything is fitted against them
    auto shape_worker = [&](std::size_t start, std::size_t end) {
      for (std::size_t i = start; i < end; ++i) {
        if (refl_flags[i] & dials::af::DontIntegrate) {
          ++n_collected;
          continue;
        }
        Shoebox<> shoebox = shoeboxes[i];
        int6 bbox = bboxes[i];

        // Corrected exactly as in the second pass, so that a shape is measured
        // on the same quantity it will later be fitted against
        Shoebox<> i_shoebox, e_shoebox;
        const Shoebox<>* i_shoebox_ptr = nullptr;
        const Shoebox<>* e_shoebox_ptr = nullptr;
        if (incident_params) {
          i_shoebox = i_shoeboxes[i];
          e_shoebox = e_shoeboxes[i];
          i_shoebox_ptr = &i_shoebox;
          e_shoebox_ptr = &e_shoebox;
        }

        ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
          shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);

        PixelCorrector corrector(
          geometry,
          apply_lorentz_correction,
          lorentz_wavelength_power,
          inputs.shoebox_pixel_count.n_signal,
          inputs.shoebox_pixel_count.n_background,
          incident_params.get_ptr(),
          absorption_params.get_ptr(),
          incident_params ? &inputs.smoothed_incident : nullptr,
          incident_params ? &inputs.smoothed_empty : nullptr,
          incident_params ? &inputs.shoebox_pixel_count.n_contrib : nullptr);

        ShoeboxIntegrationResult shoebox_result = integrate_shoebox(
          shoebox, bbox, geometry, corrector, inputs.shoebox_pixel_count.success);

        if (shoebox_result.success) {
          double cx = 0.5 * (bbox[0] + bbox[1]);
          double cy = 0.5 * (bbox[2] + bbox[3]);
          ReflectionGeometry reflection_geometry;
          if (have_predictions) {
            reflection_geometry.x_cal = xyzcal_px[i][0];
            reflection_geometry.y_cal = xyzcal_px[i][1];
            reflection_geometry.tof_cal = tof_cal[i];
            reflection_geometry.x0 = bbox[0];
            reflection_geometry.y0 = bbox[2];
          }
          profile_fitter->collect(shoebox_result,
                                  shoebox.panel,
                                  cx,
                                  cy,
                                  have_predictions ? &reflection_geometry : nullptr);
        }
        ++n_collected;
      }
    };

    auto worker = [&](std::size_t start, std::size_t end) {
      for (std::size_t i = start; i < end; ++i) {
        if (refl_flags[i] & dials::af::DontIntegrate) {
          ++n_done;
          continue;
        }
        Shoebox<> shoebox = shoeboxes[i];
        int6 bbox = bboxes[i];

        // incident and empty
        Shoebox<> i_shoebox, e_shoebox;
        const Shoebox<>* i_shoebox_ptr = nullptr;
        const Shoebox<>* e_shoebox_ptr = nullptr;
        if (incident_params) {
          i_shoebox = i_shoeboxes[i];
          e_shoebox = e_shoeboxes[i];
          i_shoebox_ptr = &i_shoebox;
          e_shoebox_ptr = &e_shoebox;
        }

        ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
          shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);

        PixelCorrector corrector(
          geometry,
          apply_lorentz_correction,
          lorentz_wavelength_power,
          inputs.shoebox_pixel_count.n_signal,
          inputs.shoebox_pixel_count.n_background,
          incident_params.get_ptr(),
          absorption_params.get_ptr(),
          incident_params ? &inputs.smoothed_incident : nullptr,
          incident_params ? &inputs.smoothed_empty : nullptr,
          incident_params ? &inputs.shoebox_pixel_count.n_contrib : nullptr);

        ShoeboxIntegrationResult shoebox_result = integrate_shoebox(
          shoebox, bbox, geometry, corrector, inputs.shoebox_pixel_count.success);

        // Overall values for shoebox summation
        succeeded[i] = shoebox_result.success;
        intensities[i] = shoebox_result.intensity;
        variances[i] = shoebox_result.variance;

        if (profile_fitter) {
          bool profile_success = false;
          double I_prf = 0.0, var_prf = 0.0;
          if (shoebox_result.success) {
            if (profile_fitter->needs_two_passes()) {
              double cx = 0.5 * (bbox[0] + bbox[1]);
              double cy = 0.5 * (bbox[2] + bbox[3]);
              profile_success = profile_fitter->fit_with_library(
                shoebox_result, shoebox.panel, cx, cy, I_prf, var_prf);
            } else {
              profile_success = profile_fitter->fit(shoebox_result, I_prf, var_prf);
            }
          }
          if (profile_success) {
            intensities_prf[i] = I_prf;
            variances_prf[i] = var_prf;
          }
          succeeded_prf[i] = profile_success;
          if (profile_success) {
            ++n_prf_ok;
          }
        }
        ++n_done;
      }
    };

    std::size_t chunk_size = (n_reflections + n_threads - 1) / n_threads;

    /*
     * Pass one, for fitters that build a profile library.  Runs to completion
     * before the second pass starts, so that every reflection sees the whole
     * library rather than however much of it happened to exist when its turn
     * came.
     */
    if (profile_fitter && profile_fitter->needs_two_passes()) {
      std::cout << "      measuring profile shapes on strong reflections" << std::endl;
      dials::util::ThreadPool shape_pool(n_threads);
      for (int t = 0; t < n_threads; ++t) {
        std::size_t start = t * chunk_size;
        std::size_t end = std::min(start + chunk_size, n_reflections);
        if (start >= end) break;
        shape_pool.post([=]() { shape_worker(start, end); });
      }
      shape_pool.wait();
      profile_fitter->finalise_library();
      std::cout << "      profile library holds " << profile_fitter->library_size()
                << " shapes from " << n_reflections << " reflections" << std::endl;
      if (have_predictions) {
        std::cout << "      " << profile_fitter->library_size_xy()
                  << " of them with the peak shape on the detector face" << std::endl;
      }
    }

    dials::util::ThreadPool pool(n_threads);

    for (int t = 0; t < n_threads; ++t) {
      std::size_t start = t * chunk_size;
      std::size_t end = std::min(start + chunk_size, n_reflections);
      if (start >= end) break;

      pool.post([=]() { worker(start, end); });
    }

    /*
     * Report progress every few seconds while the pool drains.  Profile
     * fitting a large table can take many minutes with nothing else to show
     * that it is making progress, and the running fit fraction says early on
     * whether the chosen method is working at all.
     */
    std::atomic<bool> pool_finished(false);
    std::thread reporter([&]() {
      const auto interval = std::chrono::seconds(10);
      auto next = std::chrono::steady_clock::now() + interval;
      while (!pool_finished.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (std::chrono::steady_clock::now() < next) {
          continue;
        }
        next += interval;
        std::size_t done = n_done.load();
        if (done == 0 || done >= n_reflections) {
          continue;
        }
        std::cout << "      " << done << " / " << n_reflections << " reflections ("
                  << (100 * done / n_reflections) << " %)";
        if (profile_fitter) {
          std::cout << ", profile fitted " << n_prf_ok.load() << " ("
                    << (100 * n_prf_ok.load() / done) << " % so far)";
        }
        std::cout << std::endl;
      }
    });

    pool.wait();
    pool_finished.store(true);
    reporter.join();

    reflection_table["intensity.sum.value"] = intensities;
    reflection_table["intensity.sum.variance"] = variances;

    if (profile_fitter) {
      reflection_table["intensity.prf.value"] = intensities_prf;
      reflection_table["intensity.prf.variance"] = variances_prf;
    }

    // Update flags
    dials::af::ref<std::size_t> flags =
      reflection_table.get<std::size_t>("flags").ref();
    for (std::size_t i = 0; i < flags.size(); ++i) {
      if (succeeded[i]) {
        flags[i] &= ~dials::af::FailedDuringSummation;
        flags[i] |= dials::af::IntegratedSum;
      } else {
        flags[i] &= ~dials::af::IntegratedSum;
        flags[i] |= dials::af::FailedDuringSummation;
      }
      if (profile_fitter) {
        if (succeeded_prf[i]) {
          flags[i] &= ~dials::af::FailedDuringProfileFitting;
          flags[i] |= dials::af::IntegratedPrf;
        } else {
          flags[i] &= ~dials::af::IntegratedPrf;
          flags[i] |= dials::af::FailedDuringProfileFitting;
        }
      }
    }
  }

  // Single-reflection variant used for diagnostics/plotting
  // Calculates raw_projected_intensity, projected_intensity, projected_background,
  // sum_intensity and sum_variance, with the same correction
  // combinations as integrate_reflection_table
  inline boost::python::tuple calculate_line_profile_for_reflection(
    dials::af::reflection_table& reflection,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    double lorentz_wavelength_power = 4.0) {
    // This is a slight hack to make using current interfaces easier
    // E.g. the shoebox processor
    DIALS_ASSERT(reflection.size() == 1);

    TOFGeometryContext geometry(experiment, data);

    dials::af::shared<Shoebox<>> shoeboxes = reflection["shoebox"];
    Shoebox<> shoebox = shoeboxes[0];
    int6 bbox = shoebox.bbox;

    DIALS_ASSERT(raw_projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_background_out.size() == shoebox.zsize());
    DIALS_ASSERT(tof_z_out.size() == shoebox.zsize());

    dials::af::reflection_table i_reflection_table;
    dials::af::reflection_table e_reflection_table;
    Shoebox<> i_shoebox, e_shoebox;
    const Shoebox<>* i_shoebox_ptr = nullptr;
    const Shoebox<>* e_shoebox_ptr = nullptr;
    if (incident_params) {
      i_reflection_table = load_shoebox_image_data(
        reflection, incident_params->incident_data, geometry.n_panels, data.size());
      e_reflection_table = load_shoebox_image_data(
        reflection, incident_params->empty_data, geometry.n_panels, data.size());
      dials::af::ref<Shoebox<>> i_shoeboxes = i_reflection_table["shoebox"];
      dials::af::ref<Shoebox<>> e_shoeboxes = e_reflection_table["shoebox"];
      i_shoebox = i_shoeboxes[0];
      e_shoebox = e_shoeboxes[0];
      i_shoebox_ptr = &i_shoebox;
      e_shoebox_ptr = &e_shoebox;
    }

    ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
      shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);

    PixelCorrector corrector(
      geometry,
      apply_lorentz_correction,
      lorentz_wavelength_power,
      inputs.shoebox_pixel_count.n_signal,
      inputs.shoebox_pixel_count.n_background,
      incident_params.get_ptr(),
      absorption_params.get_ptr(),
      incident_params ? &inputs.smoothed_incident : nullptr,
      incident_params ? &inputs.smoothed_empty : nullptr,
      incident_params ? &inputs.shoebox_pixel_count.n_contrib : nullptr);

    ShoeboxIntegrationResult result = integrate_shoebox(
      shoebox, bbox, geometry, corrector, inputs.shoebox_pixel_count.success);

    for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
      raw_projected_intensity_out[z] = result.raw_projected_intensity[z];
      projected_intensity_out[z] = result.projected_intensity[z];
      projected_background_out[z] = result.projected_background[z];
      tof_z_out[z] = result.tof_z[z];
    }

    return boost::python::make_tuple(result.intensity, result.variance, result.success);
  }

  // As above, but also fits a 1D profile and returns the profile-
  // fitted intensity, the line profile itself (line_profile_out), and the
  // summation results.
  inline boost::python::tuple calculate_line_profile_for_reflection(
    dials::af::reflection_table& reflection,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    scitbx::af::shared<double> line_profile_out,
    const bool& apply_lorentz_correction,
    TOFProfile1DIBIXParams& profile_params_1d_ibix,
    double lorentz_wavelength_power = 4.0) {
    boost::python::tuple result =
      calculate_line_profile_for_reflection(reflection,
                                            experiment,
                                            data,
                                            incident_params,
                                            absorption_params,
                                            raw_projected_intensity_out,
                                            projected_intensity_out,
                                            projected_background_out,
                                            tof_z_out,
                                            apply_lorentz_correction);

    double I_sum = boost::python::extract<double>(result[0]);
    double var_sum = boost::python::extract<double>(result[1]);
    bool success = boost::python::extract<bool>(result[2]);

    double I_prf = 0;
    double var_prf = 0;
    bool profile_success = false;

    if (success) {
      profile_success = fit_profile_1d_ibix(projected_intensity_out.const_ref(),
                                            tof_z_out.const_ref(),
                                            profile_params_1d_ibix,
                                            I_prf,
                                            line_profile_out,
                                            true);
    }
    return boost::python::make_tuple(I_prf, var_prf, I_sum, var_sum, profile_success);
  }

  // As above, but fits a 1D Ikeda-Carpenter profile instead of the IBIX
  // profile, returning the profile-fitted intensity, the line profile
  // itself (line_profile_out), and the summation results.
  inline boost::python::tuple calculate_line_profile_for_reflection(
    dials::af::reflection_table& reflection,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    scitbx::af::shared<double> line_profile_out,
    const bool& apply_lorentz_correction,
    TOFProfile1DICParams& profile_params_1d_ic) {
    boost::python::tuple result =
      calculate_line_profile_for_reflection(reflection,
                                            experiment,
                                            data,
                                            incident_params,
                                            absorption_params,
                                            raw_projected_intensity_out,
                                            projected_intensity_out,
                                            projected_background_out,
                                            tof_z_out,
                                            apply_lorentz_correction);

    double I_sum = boost::python::extract<double>(result[0]);
    double var_sum = boost::python::extract<double>(result[1]);
    bool success = boost::python::extract<bool>(result[2]);

    double I_prf = 0;
    double var_prf = 0;
    bool profile_success = false;

    if (success) {
      profile_success = fit_profile_1d_ic(projected_intensity_out.const_ref(),
                                          tof_z_out.const_ref(),
                                          profile_params_1d_ic,
                                          I_prf,
                                          line_profile_out,
                                          true);
    }
    return boost::python::make_tuple(I_prf, var_prf, I_sum, var_sum, profile_success);
  }

  // As calculate_line_profile_for_reflection, but also fits a 3D
  // profile and returns the profile-fitted intensity, the fitted 3D
  // profile, and the summation results.
  inline boost::python::tuple calculate_line_profile_for_reflection_3d(
    dials::af::reflection_table& reflection,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DGutmannParams& profile_params_3d_gutmann,
    double lorentz_wavelength_power = 4.0) {
    DIALS_ASSERT(reflection.size() == 1);

    TOFGeometryContext geometry(experiment, data);

    dials::af::shared<Shoebox<>> shoeboxes = reflection["shoebox"];
    Shoebox<> shoebox = shoeboxes[0];
    int6 bbox = shoebox.bbox;

    DIALS_ASSERT(raw_projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_background_out.size() == shoebox.zsize());
    DIALS_ASSERT(tof_z_out.size() == shoebox.zsize());

    dials::af::reflection_table i_reflection_table;
    dials::af::reflection_table e_reflection_table;
    Shoebox<> i_shoebox, e_shoebox;
    const Shoebox<>* i_shoebox_ptr = nullptr;
    const Shoebox<>* e_shoebox_ptr = nullptr;
    if (incident_params) {
      i_reflection_table = load_shoebox_image_data(
        reflection, incident_params->incident_data, geometry.n_panels, data.size());
      e_reflection_table = load_shoebox_image_data(
        reflection, incident_params->empty_data, geometry.n_panels, data.size());
      dials::af::ref<Shoebox<>> i_shoeboxes = i_reflection_table["shoebox"];
      dials::af::ref<Shoebox<>> e_shoeboxes = e_reflection_table["shoebox"];
      i_shoebox = i_shoeboxes[0];
      e_shoebox = e_shoeboxes[0];
      i_shoebox_ptr = &i_shoebox;
      e_shoebox_ptr = &e_shoebox;
    }

    ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
      shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);

    PixelCorrector corrector(
      geometry,
      apply_lorentz_correction,
      lorentz_wavelength_power,
      inputs.shoebox_pixel_count.n_signal,
      inputs.shoebox_pixel_count.n_background,
      incident_params.get_ptr(),
      absorption_params.get_ptr(),
      incident_params ? &inputs.smoothed_incident : nullptr,
      incident_params ? &inputs.smoothed_empty : nullptr,
      incident_params ? &inputs.shoebox_pixel_count.n_contrib : nullptr);

    ShoeboxIntegrationResult result = integrate_shoebox(
      shoebox, bbox, geometry, corrector, inputs.shoebox_pixel_count.success);

    for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
      raw_projected_intensity_out[z] = result.raw_projected_intensity[z];
      projected_intensity_out[z] = result.projected_intensity[z];
      projected_background_out[z] = result.projected_background[z];
      tof_z_out[z] = result.tof_z[z];
    }

    double I_prf = 0;
    double var_prf = 0;
    bool profile_success = false;
    scitbx::af::versa<double, scitbx::af::c_grid<3>> profile_3d_out(
      result.intensity_3d.accessor());

    if (result.success) {
      profile_success = fit_profile_3d_gutmann(result.coords_3d.const_ref(),
                                               result.intensity_3d,
                                               result.background_var_3d,
                                               profile_params_3d_gutmann,
                                               I_prf,
                                               profile_3d_out,
                                               true);
    }
    return boost::python::make_tuple(I_prf,
                                     var_prf,
                                     result.intensity,
                                     result.variance,
                                     profile_success,
                                     profile_3d_out);
  }

  // As calculate_line_profile_for_reflection_3d, but fits a 3D
  // Ikeda-Carpenter profile instead of a Gutmann profile.
  inline boost::python::tuple calculate_line_profile_for_reflection_3d(
    dials::af::reflection_table& reflection,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DICParams& profile_params_3d_ic,
    double lorentz_wavelength_power = 4.0) {
    DIALS_ASSERT(reflection.size() == 1);

    TOFGeometryContext geometry(experiment, data);

    dials::af::shared<Shoebox<>> shoeboxes = reflection["shoebox"];
    Shoebox<> shoebox = shoeboxes[0];
    int6 bbox = shoebox.bbox;

    DIALS_ASSERT(raw_projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_background_out.size() == shoebox.zsize());
    DIALS_ASSERT(tof_z_out.size() == shoebox.zsize());

    dials::af::reflection_table i_reflection_table;
    dials::af::reflection_table e_reflection_table;
    Shoebox<> i_shoebox, e_shoebox;
    const Shoebox<>* i_shoebox_ptr = nullptr;
    const Shoebox<>* e_shoebox_ptr = nullptr;
    if (incident_params) {
      i_reflection_table = load_shoebox_image_data(
        reflection, incident_params->incident_data, geometry.n_panels, data.size());
      e_reflection_table = load_shoebox_image_data(
        reflection, incident_params->empty_data, geometry.n_panels, data.size());
      dials::af::ref<Shoebox<>> i_shoeboxes = i_reflection_table["shoebox"];
      dials::af::ref<Shoebox<>> e_shoeboxes = e_reflection_table["shoebox"];
      i_shoebox = i_shoeboxes[0];
      e_shoebox = e_shoeboxes[0];
      i_shoebox_ptr = &i_shoebox;
      e_shoebox_ptr = &e_shoebox;
    }

    ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
      shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);

    PixelCorrector corrector(
      geometry,
      apply_lorentz_correction,
      lorentz_wavelength_power,
      inputs.shoebox_pixel_count.n_signal,
      inputs.shoebox_pixel_count.n_background,
      incident_params.get_ptr(),
      absorption_params.get_ptr(),
      incident_params ? &inputs.smoothed_incident : nullptr,
      incident_params ? &inputs.smoothed_empty : nullptr,
      incident_params ? &inputs.shoebox_pixel_count.n_contrib : nullptr);

    ShoeboxIntegrationResult result = integrate_shoebox(
      shoebox, bbox, geometry, corrector, inputs.shoebox_pixel_count.success);

    for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
      raw_projected_intensity_out[z] = result.raw_projected_intensity[z];
      projected_intensity_out[z] = result.projected_intensity[z];
      projected_background_out[z] = result.projected_background[z];
      tof_z_out[z] = result.tof_z[z];
    }

    double I_prf = 0;
    double var_prf = 0;
    bool profile_success = false;
    scitbx::af::versa<double, scitbx::af::c_grid<3>> profile_3d_out(
      result.intensity_3d.accessor());

    if (result.success) {
      profile_success = fit_profile_3d_ic(result.coords_3d.const_ref(),
                                          result.intensity_3d,
                                          result.background_var_3d,
                                          profile_params_3d_ic,
                                          I_prf,
                                          profile_3d_out,
                                          true);
    }
    return boost::python::make_tuple(I_prf,
                                     var_prf,
                                     result.intensity,
                                     result.variance,
                                     profile_success,
                                     profile_3d_out);
  }

  // As calculate_line_profile_for_reflection_3d, but fits a 3D
  // iBIX profile instead of a Gutmann or Ikeda-Carpenter profile.
  inline boost::python::tuple calculate_line_profile_for_reflection_3d(
    dials::af::reflection_table& reflection,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    scitbx::af::shared<double> raw_projected_intensity_out,
    scitbx::af::shared<double> projected_intensity_out,
    scitbx::af::shared<double> projected_background_out,
    scitbx::af::shared<double> tof_z_out,
    const bool& apply_lorentz_correction,
    TOFProfile3DIBIXParams& profile_params_3d_ibix,
    double lorentz_wavelength_power = 4.0) {
    DIALS_ASSERT(reflection.size() == 1);

    TOFGeometryContext geometry(experiment, data);

    dials::af::shared<Shoebox<>> shoeboxes = reflection["shoebox"];
    Shoebox<> shoebox = shoeboxes[0];
    int6 bbox = shoebox.bbox;

    DIALS_ASSERT(raw_projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_intensity_out.size() == shoebox.zsize());
    DIALS_ASSERT(projected_background_out.size() == shoebox.zsize());
    DIALS_ASSERT(tof_z_out.size() == shoebox.zsize());

    dials::af::reflection_table i_reflection_table;
    dials::af::reflection_table e_reflection_table;
    Shoebox<> i_shoebox, e_shoebox;
    const Shoebox<>* i_shoebox_ptr = nullptr;
    const Shoebox<>* e_shoebox_ptr = nullptr;
    if (incident_params) {
      i_reflection_table = load_shoebox_image_data(
        reflection, incident_params->incident_data, geometry.n_panels, data.size());
      e_reflection_table = load_shoebox_image_data(
        reflection, incident_params->empty_data, geometry.n_panels, data.size());
      dials::af::ref<Shoebox<>> i_shoeboxes = i_reflection_table["shoebox"];
      dials::af::ref<Shoebox<>> e_shoeboxes = e_reflection_table["shoebox"];
      i_shoebox = i_shoeboxes[0];
      e_shoebox = e_shoeboxes[0];
      i_shoebox_ptr = &i_shoebox;
      e_shoebox_ptr = &e_shoebox;
    }

    ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
      shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);

    PixelCorrector corrector(
      geometry,
      apply_lorentz_correction,
      lorentz_wavelength_power,
      inputs.shoebox_pixel_count.n_signal,
      inputs.shoebox_pixel_count.n_background,
      incident_params.get_ptr(),
      absorption_params.get_ptr(),
      incident_params ? &inputs.smoothed_incident : nullptr,
      incident_params ? &inputs.smoothed_empty : nullptr,
      incident_params ? &inputs.shoebox_pixel_count.n_contrib : nullptr);

    ShoeboxIntegrationResult result = integrate_shoebox(
      shoebox, bbox, geometry, corrector, inputs.shoebox_pixel_count.success);

    for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
      raw_projected_intensity_out[z] = result.raw_projected_intensity[z];
      projected_intensity_out[z] = result.projected_intensity[z];
      projected_background_out[z] = result.projected_background[z];
      tof_z_out[z] = result.tof_z[z];
    }

    double I_prf = 0;
    double var_prf = 0;
    bool profile_success = false;
    scitbx::af::versa<double, scitbx::af::c_grid<3>> profile_3d_out(
      result.intensity_3d.accessor());

    if (result.success) {
      profile_success = fit_profile_3d_ibix(result.coords_3d.const_ref(),
                                            result.intensity_3d,
                                            result.background_var_3d,
                                            profile_params_3d_ibix,
                                            I_prf,
                                            profile_3d_out,
                                            true);
    }
    return boost::python::make_tuple(I_prf,
                                     var_prf,
                                     result.intensity,
                                     result.variance,
                                     profile_success,
                                     profile_3d_out);
  }

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_INTEGRATION_H */
