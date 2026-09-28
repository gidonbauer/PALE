#pragma once

#include <cmath>

namespace Metric {

// =================================================================================================
struct Cartesian {
  static constexpr bool is_2d = true;

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h1(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq3(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq3(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h3(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh3_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh3_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto H(Float q1, Float q2) -> Float {
    return h1(q1, q2) * h2(q1, q2) * h3(q1, q2);
  }
};

// =================================================================================================
struct Polar {
  static constexpr bool is_2d = true;

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h1(Float /*q1*/, Float q2) -> Float {
    return q2;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq3(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq3(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h3(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh3_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh3_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto H(Float q1, Float q2) -> Float {
    return h1(q1, q2) * h2(q1, q2) * h3(q1, q2);
  }
};

// TODO: Metric for rotationally symmetric spherical coordinates.
//       This likely requires to re-formulate the metrics we already have in terms of cell-volume
//       metric cm and face-metrics fm1 and fm2.
struct SymmetricSpherical {
  static constexpr bool is_2d = false;

  // q1 = theta
  // q2 = r
  // q3 = phi

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h1(Float /*q1*/, Float q2) -> Float {
    return q2;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq3(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq3(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h3(Float q1, Float q2) -> Float {
    return q2 * std::sin(q1);
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh3_dq1(Float q1, Float q2) -> Float {
    return q2 * std::cos(q1);
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh3_dq2(Float q1, Float /*q2*/) -> Float {
    return std::sin(q1);
  }

  // ---------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto H(Float q1, Float q2) -> Float {
    return h1(q1, q2) * h2(q1, q2) * h3(q1, q2);
  }
};

}  // namespace Metric
