#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_OVERLAP_DECONVOLUTION_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_OVERLAP_DECONVOLUTION_H

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <dials/algorithms/integration/tof/tof_integration.h>
#include <dials/algorithms/integration/tof/tof_overlaps.h>
#include <dials/util/thread_pool.h>

/*
 * Joint fit of reflections that overlap.
 *
 * A reflection whose foreground reaches a neighbour's cannot be summed, and a
 * profile fitted to it alone would absorb part of the neighbour. But if the
 * shape and position of every peak in the cluster are known, the data are
 * linear in the intensities:
 *
 *     d_i = sum_k a_k P_ik + noise
 *
 * with P_ik the unit-intensity profile of reflection k at voxel i, and the
 * intensities a_k follow from one weighted least-squares solve. The shapes are
 * not fitted here. They come from the profile library measured on strong,
 * unoverlapped reflections nearby (ProfileFitter::local_peak_model), as a
 * PeakShape (tof_peak_shape.h): whatever model the method fits, this only
 * needs it evaluated over a box. Nor are the positions fitted: each peak is
 * placed at its prediction, and its profile broadened by the scatter of
 * observed about predicted positions that the library measured
 * (tof_profile_library.h).
 *
 *     a = (P^T W P)^-1 P^T W d,  cov(a) = (P^T W P)^-1
 *
 * with W the inverse voxel variances.
 */

namespace dials { namespace algorithms {

  /*
   * The profiles of a cluster's reflections at each of its voxels. Stored by
   * row, and sparse: a reflection contributes nothing outside its own
   * bounding box.
   */
  struct OverlapClusterDesign {
    int n_components = 0;
    std::vector<std::size_t>
      row_start;  // entries of voxel i: [row_start[i], row_start[i + 1])
    std::vector<int> component;
    std::vector<double> profile;

    OverlapClusterDesign() : row_start(1, 0) {}

    std::size_t n_voxels() const {
      return row_start.size() - 1;
    }

    void add(int k, double p) {
      component.push_back(k);
      profile.push_back(p);
    }

    void end_row() {
      row_start.push_back(component.size());
    }

    // The model at voxel i for intensities a
    double model(std::size_t i, const Eigen::VectorXd& a) const {
      double m = 0.0;
      for (std::size_t e = row_start[i]; e < row_start[i + 1]; ++e) {
        m += a[component[e]] * profile[e];
      }
      return m;
    }
  };

  struct OverlapClusterFit {
    bool success = false;
    Eigen::VectorXd value;       // intensities
    Eigen::MatrixXd covariance;  // their covariance
  };

  namespace detail {

    // Inverse of a symmetric positive definite matrix, refusing one that is
    // numerically singular
    inline bool invert_spd(const Eigen::MatrixXd& m,
                           double min_condition,
                           Eigen::MatrixXd& inverse) {
      if (m.rows() == 0 || !m.allFinite()) {
        return false;
      }
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(m);
      if (solver.info() != Eigen::Success) {
        return false;
      }
      const Eigen::VectorXd& eigenvalues = solver.eigenvalues();  // ascending
      double largest = eigenvalues[eigenvalues.size() - 1];
      if (!(largest > 0.0) || !(eigenvalues[0] > min_condition * largest)) {
        return false;
      }
      inverse = solver.eigenvectors() * eigenvalues.cwiseInverse().asDiagonal()
                * solver.eigenvectors().transpose();
      return true;
    }

  }  // namespace detail

  /*
   * One weighted solve for the intensities of a cluster.
   *
   * @param design The unit-intensity profiles
   * @param data The background-subtracted value of each voxel
   * @param variance The variance of each voxel; one that is not positive and
   * finite drops the voxel
   */
  inline OverlapClusterFit solve_overlap_cluster(const OverlapClusterDesign& design,
                                                 const std::vector<double>& data,
                                                 const std::vector<double>& variance) {
    const int K = design.n_components;
    DIALS_ASSERT(data.size() == design.n_voxels());
    DIALS_ASSERT(variance.size() == design.n_voxels());

    OverlapClusterFit fit;
    if (K == 0) {
      return fit;
    }

    // Normal equations
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(K, K);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(K);
    for (std::size_t i = 0; i < design.n_voxels(); ++i) {
      double v = variance[i];
      if (!(v > 0.0) || !std::isfinite(v) || !std::isfinite(data[i])) {
        continue;
      }
      double w = 1.0 / v;
      for (std::size_t e1 = design.row_start[i]; e1 < design.row_start[i + 1]; ++e1) {
        double wp = w * design.profile[e1];
        b[design.component[e1]] += wp * data[i];
        for (std::size_t e2 = design.row_start[i]; e2 < design.row_start[i + 1]; ++e2) {
          M(design.component[e1], design.component[e2]) += wp * design.profile[e2];
        }
      }
    }

    Eigen::MatrixXd covariance;
    if (!detail::invert_spd(M, 1e-12, covariance)) {
      return fit;
    }
    fit.value = covariance * b;
    fit.covariance = covariance;
    fit.success = fit.value.allFinite();
    return fit;
  }

  // One reflection of a cluster: where it is and what it looks like
  struct OverlapClusterComponent {
    std::size_t index = 0;  // row of the reflection table
    scitbx::af::int6 bbox;
    vec3<double> centroid;  // predicted, for voxel ownership
    // Unit-intensity profile over the bounding box, index (x * ny + y) * nz + z
    std::vector<double> values;

    bool contains(int i, int j, int k) const {
      return i >= bbox[0] && i < bbox[1] && j >= bbox[2] && j < bbox[3] && k >= bbox[4]
             && k < bbox[5];
    }

    // Unit-intensity profile at detector voxel (i, j, k), which must be inside
    // the bounding box
    double profile(int i, int j, int k) const {
      std::size_t ny = bbox[3] - bbox[2];
      std::size_t nz = bbox[5] - bbox[4];
      return values[((i - bbox[0]) * ny + (j - bbox[2])) * nz + (k - bbox[4])];
    }
  };

  /*
   * The profile of a reflection over its bounding box: the shape the library
   * expects there, placed at the predicted position corrected by the offset
   * the library expects there.
   *
   * Normalised to unit sum over the bounding box, which makes the fitted
   * intensity the intensity inside the box: the same quantity summation
   * measures. (Normalising over the whole peak instead, to extrapolate past
   * the box edge, was tried: on MANDI data it made equivalents agree less,
   * because errors in the model's tails are then multiplied up.)
   */
  inline bool make_overlap_cluster_component(std::size_t index,
                                             const scitbx::af::int6& bbox,
                                             const vec3<double>& xyzcal_px,
                                             double tof_cal,
                                             const LocalPeakModel& model,
                                             const scitbx::af::shared<double>& img_tof,
                                             OverlapClusterComponent& component) {
    if (!model.shape) {
      return false;
    }
    if (bbox[1] <= bbox[0] || bbox[3] <= bbox[2] || bbox[5] <= bbox[4] || bbox[4] < 0
        || bbox[5] > static_cast<int>(img_tof.size())) {
      return false;
    }
    component.index = index;
    component.bbox = bbox;
    component.centroid = xyzcal_px;

    const std::size_t nx = bbox[1] - bbox[0];
    const std::size_t ny = bbox[3] - bbox[2];
    const std::size_t nz = bbox[5] - bbox[4];
    std::vector<double> tof(nz);
    for (std::size_t z = 0; z < nz; ++z) {
      tof[z] = img_tof[bbox[4] + z];
    }
    PeakPosition centre;
    centre.x = xyzcal_px[0] + model.offset.x;
    centre.y = xyzcal_px[1] + model.offset.y;
    centre.t = tof_cal + model.offset.t;
    return model.shape->box_profile(
      bbox[0], bbox[2], nx, ny, tof, centre, component.values);
  }

  /*
   * Fits every cluster of overlapped reflections jointly and writes the
   * result to the table.
   *
   * Reads overlap.cluster_id (negative: not in a cluster). Writes, for every
   * member of a fitted cluster, overlap.fit.value / overlap.fit.variance
   * (variance -1 for a reflection not fitted). A member flagged OverlappedFg,
   * which could not be integrated on its own, also takes the fit as its
   * profile-fitted intensity. A member that was integrated on its own keeps
   * that intensity; its joint fit is there to be compared with it.
   *
   * Voxels: each detector voxel of the cluster is used once, from the shoebox
   * of the member that owns it, so with that member's background. A voxel
   * flagged Overlapped that lies in no other member's box was claimed by a
   * reflection outside the cluster, and is left out.
   *
   * Weights: the variance of a voxel is taken from the counts the model
   * expects there rather than the counts observed, and the solve repeated.
   * For data this sparse the observed counts are mostly zero, and weighting by
   * them would discard exactly the voxels that say a peak is absent. At
   * convergence this is the Poisson maximum-likelihood solution.
   *
   * Variance of a voxel: the Poisson variance of the counts the model expects
   * there (background plus peaks), plus whatever the incident and empty
   * spectra add. The background level is not known exactly either, but its
   * error is the same shift for every voxel of a shoebox, not independent
   * noise in each: it is added to the covariance of the intensities once, as
   * a common shift of each owner's background of variance B / n_background
   * (B its mean count per voxel), propagated through the fit.
   *
   * Model error: where the library's shape does not describe a peak, the
   * residuals over the peak exceed what counting explains. The reduced
   * chi-squared over the voxels where some member's profile reaches 5% of its
   * maximum is written to overlap.fit.chi2, and where it exceeds 1 the
   * covariance from counting is multiplied by it.
   *
   * Scale: a profile fitted in 3D and one fitted to the ToF projection do not
   * measure quite the same quantity, because they extrapolate the tails of a
   * peak differently. intensity_scale is the ratio of the first to the second,
   * as measured on reflections that can be fitted both ways. Results are
   * divided by it, so that everything in the table stays on the scale of the
   * other profile-fitted intensities.
   *
   * replace_all: every member of a cluster takes the fit as its
   * profile-fitted intensity, whether or not it was overlapped, and a member
   * of a cluster that could not be fitted loses its profile-fitted intensity.
   * Used to fit every reflection with this one model (an unoverlapped one as a
   * cluster of its own), so that all profile-fitted intensities come from the
   * same estimator.
   *
   * Returns the number of clusters fitted.
   */
  inline std::size_t tof_deconvolve_overlaps(
    dials::af::reflection_table& reflection_table,
    Experiment& experiment,
    ImageSequence& data,
    boost::optional<dials_scaling::TOFIncidentSpectrumParams> incident_params,
    boost::optional<dials_scaling::TOFAbsorptionParams> absorption_params,
    bool apply_lorentz_correction,
    double lorentz_wavelength_power,
    int n_threads,
    std::shared_ptr<ProfileFitter> profile_fitter,
    double intensity_scale = 1.0,
    bool replace_all = false,
    int n_reweight = 4,
    double min_expected_counts = 0.01) {
    DIALS_ASSERT(profile_fitter != nullptr);
    DIALS_ASSERT(n_threads > 0);
    DIALS_ASSERT(intensity_scale > 0.0);
    const std::size_t n_reflections = reflection_table.size();
    TOFGeometryContext geometry(experiment, data);

    dials::af::shared<Shoebox<> > shoeboxes = reflection_table["shoebox"];
    dials::af::const_ref<int6> bboxes = reflection_table["bbox"];
    dials::af::const_ref<vec3<double> > xyzcal_px = reflection_table["xyzcal.px"];
    dials::af::const_ref<double> tof_cal = reflection_table["tof_cal"];
    dials::af::const_ref<int> cluster_id = reflection_table["overlap.cluster_id"];
    dials::af::shared<std::size_t> refl_flags =
      reflection_table.get<std::size_t>("flags");

    // incident and empty
    dials::af::reflection_table i_reflection_table;
    dials::af::reflection_table e_reflection_table;
    dials::af::shared<Shoebox<> > i_shoeboxes;
    dials::af::shared<Shoebox<> > e_shoeboxes;
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

    // Members of each cluster
    std::map<int, std::vector<std::size_t> > by_id;
    for (std::size_t i = 0; i < n_reflections; ++i) {
      if (cluster_id[i] >= 0) {
        by_id[cluster_id[i]].push_back(i);
      }
    }
    std::vector<std::vector<std::size_t> > clusters;
    clusters.reserve(by_id.size());
    for (std::map<int, std::vector<std::size_t> >::const_iterator it = by_id.begin();
         it != by_id.end();
         ++it) {
      clusters.push_back(it->second);
    }

    dials::af::shared<double> fit_value(n_reflections, 0.0);
    dials::af::shared<double> fit_variance(n_reflections, -1.0);
    dials::af::shared<double> fit_chi2(n_reflections, -1.0);
    dials::af::shared<double> fit_fraction(n_reflections, 0.0);
    std::vector<char> cluster_fitted(clusters.size(), 0);

    auto fit_cluster = [&](const std::vector<std::size_t>& members) -> bool {
      const int K = static_cast<int>(members.size());

      // The shape and position of every member
      std::vector<OverlapClusterComponent> components(K);
      for (int a = 0; a < K; ++a) {
        std::size_t m = members[a];
        int6 bbox = bboxes[m];
        Shoebox<> shoebox = shoeboxes[m];
        LocalPeakModel peak_model;
        if (!profile_fitter->local_peak_model(shoebox.panel,
                                              0.5 * (bbox[0] + bbox[1]),
                                              0.5 * (bbox[2] + bbox[3]),
                                              peak_model)
            || !make_overlap_cluster_component(m,
                                               bbox,
                                               xyzcal_px[m],
                                               tof_cal[m],
                                               peak_model,
                                               geometry.img_tof,
                                               components[a])) {
          return false;
        }
      }

      // The voxels, each once, from the shoebox of its owner
      OverlapClusterDesign design;
      design.n_components = K;
      std::vector<double> voxel_data;
      std::vector<double> voxel_gain;        // corrected value of one count
      std::vector<double> voxel_background;  // counts
      std::vector<double> voxel_fixed_var;   // variance not from the counts themselves
      std::vector<int> voxel_owner;          // member whose shoebox the voxel is from
      std::vector<double> owner_background(K, 0.0);  // summed background counts
      std::vector<double> owner_n_voxels(K, 0.0);
      std::vector<double> owner_n_background(K, 0.0);
      double total_counts = 0.0;

      for (int a = 0; a < K; ++a) {
        std::size_t m = members[a];
        Shoebox<> shoebox = shoeboxes[m];
        if (!shoebox.is_consistent()) {
          return false;
        }
        const OverlapClusterComponent& ca = components[a];
        const int6& bbox = ca.bbox;
        int panel = shoebox.panel;

        Shoebox<> i_shoebox, e_shoebox;
        const Shoebox<>* i_shoebox_ptr = nullptr;
        const Shoebox<>* e_shoebox_ptr = nullptr;
        if (incident_params) {
          i_shoebox = i_shoeboxes[m];
          e_shoebox = e_shoeboxes[m];
          i_shoebox_ptr = &i_shoebox;
          e_shoebox_ptr = &e_shoebox;
        }
        ShoeboxCorrectorInputs inputs = prepare_shoebox_corrector_inputs(
          shoebox, bbox, geometry.image_size, i_shoebox_ptr, e_shoebox_ptr);
        const double n_sig = inputs.shoebox_pixel_count.n_signal;
        const double n_bg = inputs.shoebox_pixel_count.n_background;
        // the factor PixelCorrector applies to the background's variance
        const double background_var_factor = n_bg > 0 ? 1.0 + n_sig / n_bg : 1.0;
        owner_n_background[a] = n_bg;
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

        for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
          int frame_z = bbox[4] + z;
          if (frame_z < 1) {
            continue;  // no bin width for the first frame
          }
          double tof = geometry.img_tof[frame_z];
          for (std::size_t y = 0; y < shoebox.ysize(); ++y) {
            int panel_y = bbox[2] + y;
            if (panel_y >= static_cast<int>(geometry.image_size[1]) || panel_y < 0) {
              continue;
            }
            for (std::size_t x = 0; x < shoebox.xsize(); ++x) {
              int panel_x = bbox[0] + x;
              if (panel_x >= static_cast<int>(geometry.image_size[0]) || panel_x < 0) {
                continue;
              }
              int mask = shoebox.mask(z, y, x);
              if ((mask & Valid) == 0) {
                continue;
              }

              // Only from its owner, and not if a reflection outside the
              // cluster has a claim on it
              bool owner = true;
              bool in_other = false;
              for (int c = 0; c < K && owner; ++c) {
                if (c == a || !components[c].contains(panel_x, panel_y, frame_z)) {
                  continue;
                }
                in_other = true;
                owner = tof_voxel_owned_by_first(bbox,
                                                 ca.centroid,
                                                 ca.index,
                                                 components[c].bbox,
                                                 components[c].centroid,
                                                 components[c].index,
                                                 panel_x,
                                                 panel_y,
                                                 frame_z);
              }
              if (!owner || ((mask & Overlapped) != 0 && !in_other)) {
                continue;
              }

              double raw_S = shoebox.data(z, y, x);
              double raw_B = shoebox.background(z, y, x);
              PixelCorrection pixel, pixel_plus_one;
              if (!corrector.correct(
                    panel, panel_x, panel_y, frame_z, z, tof, raw_S, raw_B, pixel)
                  || !corrector.correct(panel,
                                        panel_x,
                                        panel_y,
                                        frame_z,
                                        z,
                                        tof,
                                        raw_S + 1.0,
                                        raw_B,
                                        pixel_plus_one)) {
                continue;
              }
              double gain = pixel_plus_one.I - pixel.I;
              if (!(gain > 0.0) || !std::isfinite(gain) || !std::isfinite(pixel.I)
                  || !std::isfinite(pixel.var_I)) {
                continue;
              }

              for (int c = 0; c < K; ++c) {
                if (components[c].contains(panel_x, panel_y, frame_z)) {
                  double p = components[c].profile(panel_x, panel_y, frame_z);
                  if (p > 0.0) {
                    design.add(c, p);
                  }
                }
              }
              design.end_row();
              voxel_data.push_back(pixel.I);
              voxel_gain.push_back(gain);
              voxel_background.push_back(std::max(raw_B, 0.0));
              // var_I = gain^2 (|raw_S| + |raw_B| background_var_factor) +
              // the incident and empty terms. The counts are replaced by
              // what the model expects, and the background's error is
              // added separately, so only the last part is kept here
              voxel_fixed_var.push_back(std::max(
                pixel.var_I
                  - gain * gain
                      * (std::abs(raw_S) + std::abs(raw_B) * background_var_factor),
                0.0));
              voxel_owner.push_back(a);
              owner_background[a] += std::max(raw_B, 0.0);
              owner_n_voxels[a] += 1.0;
              total_counts += std::abs(raw_S);
            }
          }
        }
      }

      const std::size_t n_voxels = design.n_voxels();
      if (n_voxels < static_cast<std::size_t>(K)) {
        return false;
      }

      // How much of each box-normalised profile the voxels used hold: less than
      // 1 where voxels are masked or owned by a reflection outside the cluster
      std::vector<double> fraction(K, 0.0);
      for (std::size_t e = 0; e < design.profile.size(); ++e) {
        fraction[design.component[e]] += design.profile[e];
      }
      for (int a = 0; a < K; ++a) {
        fit_fraction[members[a]] = fraction[a];
      }

      // Start from uniform expected counts, which is an unweighted fit where
      // the gain is constant, then reweight by what the model expects
      std::vector<double> variance(n_voxels);
      double mean_counts = std::max(total_counts / n_voxels, min_expected_counts);
      for (std::size_t i = 0; i < n_voxels; ++i) {
        variance[i] = voxel_gain[i] * voxel_gain[i] * mean_counts + voxel_fixed_var[i];
      }
      OverlapClusterFit fit = solve_overlap_cluster(design, voxel_data, variance);
      for (int iteration = 0; iteration < n_reweight && fit.success; ++iteration) {
        for (std::size_t i = 0; i < n_voxels; ++i) {
          double expected = voxel_background[i]
                            + std::max(design.model(i, fit.value), 0.0) / voxel_gain[i];
          expected = std::max(expected, min_expected_counts);
          variance[i] = voxel_gain[i] * voxel_gain[i] * expected + voxel_fixed_var[i];
        }
        fit = solve_overlap_cluster(design, voxel_data, variance);
      }
      if (!fit.success) {
        return false;
      }

      // Model error: reduced chi-squared over the peaks
      std::vector<double> component_max(K, 0.0);
      for (std::size_t i = 0; i < n_voxels; ++i) {
        for (std::size_t e = design.row_start[i]; e < design.row_start[i + 1]; ++e) {
          component_max[design.component[e]] =
            std::max(component_max[design.component[e]], design.profile[e]);
        }
      }
      double chi2 = 0.0;
      std::size_t n_peak = 0;
      for (std::size_t i = 0; i < n_voxels; ++i) {
        bool in_peak = false;
        for (std::size_t e = design.row_start[i]; e < design.row_start[i + 1]; ++e) {
          if (design.profile[e] >= 0.05 * component_max[design.component[e]]) {
            in_peak = true;
          }
        }
        if (!in_peak || !(variance[i] > 0.0)) {
          continue;
        }
        double r = voxel_data[i] - design.model(i, fit.value);
        chi2 += r * r / variance[i];
        ++n_peak;
      }
      double reduced_chi2 = -1.0;
      double inflation = 1.0;
      if (n_peak > static_cast<std::size_t>(K)) {
        reduced_chi2 = chi2 / (n_peak - K);
        inflation = std::max(reduced_chi2, 1.0);
      }
      fit.covariance *= inflation;

      // The background level of each owner, shifted by the same amount in
      // every voxel it owns
      std::vector<Eigen::VectorXd> shift(K, Eigen::VectorXd::Zero(K));
      for (std::size_t i = 0; i < n_voxels; ++i) {
        if (!(variance[i] > 0.0)) {
          continue;
        }
        double w = voxel_gain[i] / variance[i];
        for (std::size_t e = design.row_start[i]; e < design.row_start[i + 1]; ++e) {
          shift[voxel_owner[i]][design.component[e]] += w * design.profile[e];
        }
      }
      // Covariance before inflation maps P^T W (shift) to the change in a
      Eigen::MatrixXd shift_map = fit.covariance / inflation;
      for (int o = 0; o < K; ++o) {
        if (!(owner_n_background[o] > 0.0) || !(owner_n_voxels[o] > 0.0)) {
          continue;
        }
        double var_level =
          owner_background[o] / owner_n_voxels[o] / owner_n_background[o];
        Eigen::VectorXd d = shift_map * shift[o];
        fit.covariance += var_level * d * d.transpose();
      }

      for (int a = 0; a < K; ++a) {
        if (!(fit.covariance(a, a) > 0.0) || !std::isfinite(fit.covariance(a, a))) {
          return false;
        }
      }

      for (int a = 0; a < K; ++a) {
        std::size_t m = members[a];
        fit_chi2[m] = reduced_chi2;
        fit_value[m] = fit.value[a] / intensity_scale;
        fit_variance[m] = fit.covariance(a, a) / (intensity_scale * intensity_scale);
      }
      return true;
    };

    auto worker = [&](std::size_t start, std::size_t end) {
      for (std::size_t c = start; c < end; ++c) {
        cluster_fitted[c] = fit_cluster(clusters[c]) ? 1 : 0;
      }
    };

    if (!clusters.empty()) {
      dials::util::ThreadPool pool(n_threads);
      std::size_t chunk_size = (clusters.size() + n_threads - 1) / n_threads;
      for (int t = 0; t < n_threads; ++t) {
        std::size_t start = t * chunk_size;
        std::size_t end = std::min(start + chunk_size, clusters.size());
        if (start >= end) break;
        pool.post([=]() { worker(start, end); });
      }
      pool.wait();
    }

    reflection_table["overlap.fit.value"] = fit_value;
    reflection_table["overlap.fit.variance"] = fit_variance;
    reflection_table["overlap.fit.chi2"] = fit_chi2;
    reflection_table["overlap.fit.fraction"] = fit_fraction;

    // An overlapped reflection (any member, with replace_all) takes the fit
    // as its profile-fitted intensity
    dials::af::shared<double> intensities_prf(n_reflections, 0.0);
    dials::af::shared<double> variances_prf(n_reflections, 0.0);
    if (reflection_table.contains("intensity.prf.value")
        && reflection_table.contains("intensity.prf.variance")) {
      intensities_prf = reflection_table["intensity.prf.value"];
      variances_prf = reflection_table["intensity.prf.variance"];
    }
    dials::af::ref<std::size_t> flags =
      reflection_table.get<std::size_t>("flags").ref();
    std::size_t n_fitted = 0;
    for (std::size_t c = 0; c < clusters.size(); ++c) {
      if (!cluster_fitted[c]) {
        if (replace_all) {
          for (std::size_t j = 0; j < clusters[c].size(); ++j) {
            std::size_t m = clusters[c][j];
            intensities_prf[m] = 0.0;
            variances_prf[m] = 0.0;
            flags[m] &= ~dials::af::IntegratedPrf;
            flags[m] |= dials::af::FailedDuringProfileFitting;
          }
        }
        continue;
      }
      ++n_fitted;
      for (std::size_t j = 0; j < clusters[c].size(); ++j) {
        std::size_t m = clusters[c][j];
        if ((!replace_all && (flags[m] & dials::af::OverlappedFg) == 0)
            || (flags[m] & dials::af::DontIntegrate) != 0) {
          continue;
        }
        intensities_prf[m] = fit_value[m];
        variances_prf[m] = fit_variance[m];
        flags[m] &= ~dials::af::FailedDuringProfileFitting;
        flags[m] |= dials::af::IntegratedPrf;
      }
    }
    reflection_table["intensity.prf.value"] = intensities_prf;
    reflection_table["intensity.prf.variance"] = variances_prf;
    return n_fitted;
  }

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_OVERLAP_DECONVOLUTION_H */
