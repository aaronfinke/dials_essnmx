#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_LIBRARY_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_LIBRARY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <vector>
#include <dials/algorithms/integration/tof/tof_peak_shape.h>

/*
 * A library of peak shapes measured on strong reflections, for reflections
 * that cannot determine their own.
 *
 * Each entry is a shape fitted to one reflection, kept against where on the
 * detector it was measured, with how far its peak sat from the prediction
 * where that is known. What a shape is depends on the method (the Shape
 * parameter); what is done with the offsets does not:
 *
 * The offset of one library reflection from its prediction is mostly that
 * reflection's own prediction error, not something its neighbours share: on
 * MANDI the scatter between reflections is 0.7 px and 0.7 ToF frames while
 * panel medians differ by 0.1 to 0.2. So what a neighbour can be told is the
 * typical offset and how uncertain its own position is, and both are taken
 * from robust statistics rather than from any single entry: the median offset
 * of the panel, where it holds enough entries, or of the whole detector
 * otherwise, and the scatter of entries about the median that applies to
 * them.
 */

namespace dials { namespace algorithms {

  namespace profile_library_detail {

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

  }  // namespace profile_library_detail

  template <typename Shape>
  class ProfileLibrary {
  public:
    struct Entry {
      int panel = 0;
      double x = 0.0;  // where it was measured (px)
      double y = 0.0;
      Shape shape;
      bool has_offset = false;
      PeakPosition offset;  // observed minus predicted position (px, px, us)
    };

    explicit ProfileLibrary(std::size_t min_entries_per_panel = 10)
        : min_entries_per_panel_(min_entries_per_panel) {}

    // Thread safe
    void add(const Entry& entry) {
      std::lock_guard<std::mutex> lock(mutex_);
      entries_.push_back(entry);
    }

    /*
     * Called once, when every entry has been added: the median offset per
     * panel, where it holds min_entries_per_panel entries with an offset, and
     * over the whole detector, and the scatter of each entry about the median
     * that applies to its panel.
     */
    void finalise() {
      namespace detail = profile_library_detail;
      std::lock_guard<std::mutex> lock(mutex_);
      finalised_ = false;
      panel_groups_.clear();
      panel_offsets_.clear();
      with_offset_.clear();

      for (const auto& e : entries_) {
        if (e.has_offset) {
          panel_groups_[e.panel].push_back(&e);
          with_offset_.push_back(&e);
        }
      }
      if (with_offset_.empty()) {
        panel_groups_.clear();
        return;
      }
      global_offset_ = median_offset(with_offset_);
      for (auto it = panel_groups_.begin(); it != panel_groups_.end();) {
        if (it->second.size() >= min_entries_per_panel_) {
          panel_offsets_[it->first] = median_offset(it->second);
          ++it;
        } else {
          it = panel_groups_.erase(it);
        }
      }

      std::vector<double> rx, ry, rt;
      for (const Entry* e : with_offset_) {
        const PeakPosition& o = offset(e->panel);
        rx.push_back(e->offset.x - o.x);
        ry.push_back(e->offset.y - o.y);
        rt.push_back(e->offset.t - o.t);
      }
      scatter_.x = detail::robust_sd(rx);
      scatter_.y = detail::robust_sd(ry);
      scatter_.t = detail::robust_sd(rt);
      finalised_ = true;
    }

    // Whether finalise found entries with offsets
    bool finalised() const {
      return finalised_;
    }

    std::size_t size() const {
      return entries_.size();
    }

    std::size_t size_with_offset() const {
      std::size_t n = 0;
      for (const auto& e : entries_) {
        if (e.has_offset) {
          ++n;
        }
      }
      return n;
    }

    // The entry measured nearest to (x, y), on the same panel where the panel
    // holds any; with require_offset, only entries with an offset
    const Entry* nearest(int panel,
                         double x,
                         double y,
                         bool require_offset = false) const {
      const Entry* best = nullptr;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (int same_panel = 1; same_panel >= 0; --same_panel) {
        for (const auto& e : entries_) {
          if (same_panel && e.panel != panel) {
            continue;
          }
          if (require_offset && !e.has_offset) {
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

    // The median offset that applies on a panel (set by finalise)
    const PeakPosition& offset(int panel) const {
      typename std::map<int, PeakPosition>::const_iterator it =
        panel_offsets_.find(panel);
      return it != panel_offsets_.end() ? it->second : global_offset_;
    }

    // Scatter of individual offsets about the medians (set by finalise)
    const PeakPosition& scatter() const {
      return scatter_;
    }

    /*
     * The entries the median offset of a panel is taken from: the panel's
     * own, if it holds enough, or every entry with an offset. For summaries
     * of other properties by the same rule.
     */
    const std::vector<const Entry*>& group(int panel) const {
      typename std::map<int, std::vector<const Entry*> >::const_iterator it =
        panel_groups_.find(panel);
      return it != panel_groups_.end() ? it->second : with_offset_;
    }

    // Every entry with an offset (set by finalise)
    const std::vector<const Entry*>& with_offset() const {
      return with_offset_;
    }

    // The panels with a group of their own
    std::vector<int> panels_with_group() const {
      std::vector<int> panels;
      for (const auto& item : panel_groups_) {
        panels.push_back(item.first);
      }
      return panels;
    }

  private:
    static PeakPosition median_offset(const std::vector<const Entry*>& entries) {
      namespace detail = profile_library_detail;
      std::vector<double> dx, dy, dt;
      for (const Entry* e : entries) {
        dx.push_back(e->offset.x);
        dy.push_back(e->offset.y);
        dt.push_back(e->offset.t);
      }
      PeakPosition o;
      o.x = detail::median(dx);
      o.y = detail::median(dy);
      o.t = detail::median(dt);
      return o;
    }

    std::size_t min_entries_per_panel_;
    std::vector<Entry> entries_;
    mutable std::mutex mutex_;

    // Set by finalise
    bool finalised_ = false;
    std::vector<const Entry*> with_offset_;
    std::map<int, std::vector<const Entry*> > panel_groups_;
    std::map<int, PeakPosition> panel_offsets_;
    PeakPosition global_offset_;
    PeakPosition scatter_;
  };

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_LIBRARY_H */
