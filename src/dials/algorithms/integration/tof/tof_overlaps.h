#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_OVERLAPS_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_OVERLAPS_H

#include <algorithm>
#include <utility>
#include <vector>
#include <scitbx/vec3.h>
#include <scitbx/array_family/tiny_types.h>
#include <dials/array_family/reflection_table.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/model/data/shoebox.h>
#include <dials/model/data/mask_code.h>
#include <dials/algorithms/shoebox/find_overlapping.h>
#include <dials/error.h>

/*
 * Handling of reflections whose bounding boxes overlap, as an alternative to
 * rejecting every one of them.
 *
 * Most overlaps are between the padding of one box and the padding of another,
 * and contaminate neither peak. Two steps separate those from the overlaps
 * that matter:
 *
 *  1. Before the foreground masks are calculated, every voxel shared between
 *     two boxes is given to the reflection whose predicted centroid is nearer,
 *     and flagged Overlapped in the other. A data-driven mask can then no
 *     longer be captured by a stronger neighbour sitting in its padding.
 *  2. After the foreground masks are calculated, any voxel that is foreground
 *     for one reflection is flagged Overlapped in every other box containing
 *     it, so that it is used neither as background nor in a projection there.
 *
 * A reflection whose own foreground ends up holding an Overlapped voxel is
 * really overlapped. The summation code already refuses such a reflection,
 * and the background code already ignores Overlapped voxels.
 */

namespace dials { namespace algorithms {

  /*
   * Squared distance of the centre of voxel (i, j, k) from a reflection's
   * predicted centroid, each axis in units of the half-extent of the
   * reflection's bounding box. Pixels and ToF frames are not commensurate, and
   * a box is typically many more frames deep than it is pixels wide, so an
   * unnormalised distance would let the ToF axis decide every case.
   */
  inline double tof_voxel_distance_sq(const scitbx::af::int6& bbox,
                                      const scitbx::vec3<double>& centroid,
                                      int i,
                                      int j,
                                      int k) {
    double hx = std::max(0.5 * (bbox[1] - bbox[0]), 0.5);
    double hy = std::max(0.5 * (bbox[3] - bbox[2]), 0.5);
    double hz = std::max(0.5 * (bbox[5] - bbox[4]), 0.5);
    double dx = (i + 0.5 - centroid[0]) / hx;
    double dy = (j + 0.5 - centroid[1]) / hy;
    double dz = (k + 0.5 - centroid[2]) / hz;
    return dx * dx + dy * dy + dz * dz;
  }

  /*
   * Whether reflection a, rather than reflection b, owns voxel (i, j, k).
   * Ties go to the lower index so that the answer does not depend on the order
   * the pair is presented in.
   */
  inline bool tof_voxel_owned_by_first(const scitbx::af::int6& bbox_a,
                                       const scitbx::vec3<double>& centroid_a,
                                       std::size_t index_a,
                                       const scitbx::af::int6& bbox_b,
                                       const scitbx::vec3<double>& centroid_b,
                                       std::size_t index_b,
                                       int i,
                                       int j,
                                       int k) {
    double da = tof_voxel_distance_sq(bbox_a, centroid_a, i, j, k);
    double db = tof_voxel_distance_sq(bbox_b, centroid_b, i, j, k);
    if (da != db) {
      return da < db;
    }
    return index_a < index_b;
  }

  // The voxel range two bounding boxes share. Empty if any extent is not positive.
  inline scitbx::af::int6 tof_bbox_intersection(const scitbx::af::int6& a,
                                                const scitbx::af::int6& b) {
    return scitbx::af::int6(std::max(a[0], b[0]),
                            std::min(a[1], b[1]),
                            std::max(a[2], b[2]),
                            std::min(a[3], b[3]),
                            std::max(a[4], b[4]),
                            std::min(a[5], b[5]));
  }

  // Every pair (a, b), a < b, of reflections on the same panel whose bounding
  // boxes overlap
  inline std::vector<std::pair<std::size_t, std::size_t> > tof_find_overlap_pairs(
    dials::af::reflection_table& reflection_table) {
    std::vector<std::pair<std::size_t, std::size_t> > pairs;
    if (reflection_table.size() == 0) {
      return pairs;
    }
    dials::af::const_ref<scitbx::af::int6> bboxes = reflection_table["bbox"];
    dials::af::const_ref<std::size_t> panels = reflection_table["panel"];
    dials::model::AdjacencyList list =
      shoebox::find_overlapping_multi_panel(bboxes, panels);
    dials::model::AdjacencyList::edge_iterator_range range = list.edges();
    for (dials::model::AdjacencyList::edge_iterator it = range.first;
         it != range.second;
         ++it) {
      if (it->first < it->second) {
        pairs.push_back(*it);
      }
    }
    return pairs;
  }

  /*
   * Step 1. Gives each voxel shared between two bounding boxes to the
   * reflection whose predicted centroid is nearer, by setting Overlapped on
   * the other's copy of it. Must run before the foreground masks are
   * calculated.
   *
   * Returns, per reflection, whether its bounding box overlaps any other.
   */
  inline dials::af::shared<bool> tof_assign_overlap_ownership(
    dials::af::reflection_table& reflection_table) {
    using dials::model::Overlapped;
    using dials::model::Shoebox;

    dials::af::shared<bool> has_overlap(reflection_table.size(), false);
    std::vector<std::pair<std::size_t, std::size_t> > pairs =
      tof_find_overlap_pairs(reflection_table);
    if (pairs.empty()) {
      return has_overlap;
    }

    dials::af::shared<Shoebox<> > shoeboxes = reflection_table["shoebox"];
    dials::af::const_ref<scitbx::vec3<double> > centroids =
      reflection_table["xyzcal.px"];

    for (std::size_t p = 0; p < pairs.size(); ++p) {
      std::size_t a = pairs[p].first;
      std::size_t b = pairs[p].second;
      has_overlap[a] = true;
      has_overlap[b] = true;

      Shoebox<> sbox_a = shoeboxes[a];
      Shoebox<> sbox_b = shoeboxes[b];
      const scitbx::af::int6 bbox_a = sbox_a.bbox;
      const scitbx::af::int6 bbox_b = sbox_b.bbox;
      af::ref<uint8_t, af::c_grid<3> > mask_a = sbox_a.mask.ref();
      af::ref<uint8_t, af::c_grid<3> > mask_b = sbox_b.mask.ref();
      const scitbx::af::int6 r = tof_bbox_intersection(bbox_a, bbox_b);

      for (int k = r[4]; k < r[5]; ++k) {
        for (int j = r[2]; j < r[3]; ++j) {
          for (int i = r[0]; i < r[1]; ++i) {
            if (tof_voxel_owned_by_first(
                  bbox_a, centroids[a], a, bbox_b, centroids[b], b, i, j, k)) {
              mask_b(k - bbox_b[4], j - bbox_b[2], i - bbox_b[0]) |= Overlapped;
            } else {
              mask_a(k - bbox_a[4], j - bbox_a[2], i - bbox_a[0]) |= Overlapped;
            }
          }
        }
      }
    }
    return has_overlap;
  }

  /*
   * Step 2. Flags as Overlapped, in every box that contains it, each voxel
   * that is foreground for another reflection. Must run after the foreground
   * masks are calculated.
   *
   * Returns the coupled pairs: those where the foreground of one reflection
   * reaches the foreground of the other, or voxels the other owns. These are
   * the pairs whose intensities cannot be measured independently, and their
   * connected components are the clusters to fit jointly.
   */
  inline std::vector<std::pair<std::size_t, std::size_t> >
  tof_mask_neighbour_foreground(dials::af::reflection_table& reflection_table) {
    using dials::model::Foreground;
    using dials::model::Overlapped;
    using dials::model::Shoebox;

    std::vector<std::pair<std::size_t, std::size_t> > coupled;
    std::vector<std::pair<std::size_t, std::size_t> > pairs =
      tof_find_overlap_pairs(reflection_table);
    if (pairs.empty()) {
      return coupled;
    }

    dials::af::shared<Shoebox<> > shoeboxes = reflection_table["shoebox"];
    dials::af::const_ref<scitbx::vec3<double> > centroids =
      reflection_table["xyzcal.px"];

    for (std::size_t p = 0; p < pairs.size(); ++p) {
      std::size_t a = pairs[p].first;
      std::size_t b = pairs[p].second;

      Shoebox<> sbox_a = shoeboxes[a];
      Shoebox<> sbox_b = shoeboxes[b];
      const scitbx::af::int6 bbox_a = sbox_a.bbox;
      const scitbx::af::int6 bbox_b = sbox_b.bbox;
      af::ref<uint8_t, af::c_grid<3> > mask_a = sbox_a.mask.ref();
      af::ref<uint8_t, af::c_grid<3> > mask_b = sbox_b.mask.ref();
      const scitbx::af::int6 r = tof_bbox_intersection(bbox_a, bbox_b);

      bool is_coupled = false;
      for (int k = r[4]; k < r[5]; ++k) {
        for (int j = r[2]; j < r[3]; ++j) {
          for (int i = r[0]; i < r[1]; ++i) {
            uint8_t& m_a = mask_a(k - bbox_a[4], j - bbox_a[2], i - bbox_a[0]);
            uint8_t& m_b = mask_b(k - bbox_b[4], j - bbox_b[2], i - bbox_b[0]);
            // Foreground is never modified here, so the outcome does not
            // depend on the order the pairs are visited in
            bool fg_a = (m_a & Foreground) != 0;
            bool fg_b = (m_b & Foreground) != 0;
            if (!fg_a && !fg_b) {
              continue;
            }
            if (fg_b) {
              m_a |= Overlapped;
            }
            if (fg_a) {
              m_b |= Overlapped;
            }
            if (!is_coupled) {
              if (fg_a && fg_b) {
                is_coupled = true;
              } else {
                bool a_owns = tof_voxel_owned_by_first(
                  bbox_a, centroids[a], a, bbox_b, centroids[b], b, i, j, k);
                is_coupled = (fg_a && !a_owns) || (fg_b && a_owns);
              }
            }
          }
        }
      }
      if (is_coupled) {
        coupled.push_back(pairs[p]);
      }
    }
    return coupled;
  }

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_OVERLAPS_H */
