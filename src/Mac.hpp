#pragma once

#include <Igor/Math.hpp>

#include "Grid.hpp"

#include "MacOrthogonal.hpp"

struct CartesianMetric {
  // -----------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h1(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }

  // -----------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }
};

struct PolarMetric {
  // -----------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h1(Float /*q1*/, Float q2) -> Float {
    return q2;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh1_dq2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  // -----------------------------------------------
  template <typename Float>
  [[nodiscard]] static constexpr auto h2(Float /*q1*/, Float /*q2*/) -> Float {
    return 1.0;
  }

  template <typename Float>
  [[nodiscard]] static constexpr auto dh2_dq1(Float /*q1*/, Float /*q2*/) -> Float {
    return 0.0;
  }
};

// TODO: Metric for rotationally symmetric spherical coordinates.
//       This likely requires to re-formulate the metrics we already have in terms of cell-volume
//       metric cm and face-metrics fm1 and fm2.
// struct SphericalMetric {};

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_div(const Grid<Float, LAYOUT>& grid,
                        const FaceVector<Float, LAYOUT> uf,
                        Scalar<Float, LAYOUT> div) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return OrthogonalCoordinates::calc_div<CartesianMetric>(grid, uf, div);
    case Coordinates::POLAR: return OrthogonalCoordinates::calc_div<PolarMetric>(grid, uf, div);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_mom_flux(const Grid<Float, LAYOUT>& grid,
                             const FaceVector<Float, LAYOUT> u,
                             const Scalar<Float, LAYOUT> p,
                             Float rho,
                             Float mu,
                             Scalar<Float, LAYOUT> FUX,
                             VertexScalar<Float, LAYOUT> FUY,
                             VertexScalar<Float, LAYOUT> FVX,
                             Scalar<Float, LAYOUT> FVY) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return OrthogonalCoordinates::calc_mom_flux<CartesianMetric>(
          grid, u, p, rho, mu, FUX, FUY, FVX, FVY);
    case Coordinates::POLAR:
      return OrthogonalCoordinates::calc_mom_flux<PolarMetric>(
          grid, u, p, rho, mu, FUX, FUY, FVX, FVY);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return OrthogonalCoordinates::update_u<CartesianMetric>(
          grid, dt, FUX, FUY, FVX, FVY, u_old, u);
    case Coordinates::POLAR:
      return OrthogonalCoordinates::update_u<PolarMetric>(grid, dt, FUX, FUY, FVX, FVY, u_old, u);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void correct_velocity(const Grid<Float, LAYOUT>& grid,
                                const Scalar<Float, LAYOUT> dp,
                                Float rho,
                                Float dt,
                                FaceVector<Float, LAYOUT> u,
                                Scalar<Float, LAYOUT> p) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return OrthogonalCoordinates::correct_velocity<CartesianMetric>(grid, dp, rho, dt, u, p);
    case Coordinates::POLAR:
      return OrthogonalCoordinates::correct_velocity<PolarMetric>(grid, dp, rho, dt, u, p);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr auto adjust_dt(const Grid<Float, LAYOUT>& grid,
                         const FaceVector<Float, LAYOUT> u,
                         Float rho,
                         Float mu,
                         Float CFL) noexcept -> Float {
  Float ux_max = grid.template transform_reduce_face_i<Dimension::X>(
      0.0,
      FOREACH_FUNC { return std::abs(u.x(i, j)); },
      [](Float lhs, Float rhs) { return std::max(lhs, rhs); });
  Float uy_max = grid.template transform_reduce_face_i<Dimension::Y>(
      0.0,
      FOREACH_FUNC { return std::abs(u.y(i, j)); },
      [](Float lhs, Float rhs) { return std::max(lhs, rhs); });

  // Correction for polar coordinates
  const auto hx = grid.coords() == Coordinates::POLAR ? grid.ym(0) * grid.dx() : grid.dx();
  const auto hy = grid.dy();

  // Advection: dt * (|u|/hx + |v|/hy) <= CFL
  const auto adv = ux_max / hx + uy_max / hy;
  // Diffusion: dt * 2 * nu * (1/hx^2 + 1/hy^2) <= CFL
  const auto diff         = 2.0 * (mu / rho) * (1.0 / Igor::sqr(hx) + 1.0 / Igor::sqr(hy));

  constexpr auto no_limit = std::numeric_limits<Float>::max();
  return std::min(adv > 0.0 ? CFL / adv : no_limit,  //
                  diff > 0.0 ? CFL / diff : no_limit);
}
