#pragma once

#include <Igor/Math.hpp>

#include "Grid.hpp"
#include "MacOrthogonal.hpp"
#include "Metrics.hpp"

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_div(const Grid<Float, LAYOUT>& grid,
                        const FaceVector<Float, LAYOUT> uf,
                        Scalar<Float, LAYOUT> div) {
  dispatch_metric(grid, DISPATCH_FUNC { OrthogonalCoordinates::calc_div<Metric>(grid, uf, div); });
}

// =================================================================================================
template <typename Float,
          Layout LAYOUT,
          IsNoneOr<Vec2<Float>> W_t,
          IsNoneOr<Scalar<Float, LAYOUT>> FWZ_t>
constexpr void calc_mom_flux(const Grid<Float, LAYOUT>& grid,
                             const FaceVector<Float, LAYOUT> u,
                             const Scalar<Float, LAYOUT> p,
                             Float rho,
                             Float mu,
                             const W_t& w,
                             Scalar<Float, LAYOUT> FUX,
                             VertexScalar<Float, LAYOUT> FUY,
                             VertexScalar<Float, LAYOUT> FVX,
                             Scalar<Float, LAYOUT> FVY,
                             FWZ_t FWZ) {
  dispatch_metric(
      grid, DISPATCH_FUNC {
        if constexpr (Metric::is_2d != IsNone<FWZ_t>) {
          Igor::Panic("`FWZ` must be given iff coordinates are quasi-3D");
        } else {
          OrthogonalCoordinates::calc_mom_flux<Metric>(
              grid, u, p, rho, mu, w, FUX, FUY, FVX, FVY, FWZ);
        }
      });
}

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
  return calc_mom_flux(grid, u, p, rho, mu, None{}, FUX, FUY, FVX, FVY, None{});
}

template <typename Float, Layout LAYOUT>
constexpr void calc_mom_flux(const Grid<Float, LAYOUT>& grid,
                             const FaceVector<Float, LAYOUT> u,
                             const Scalar<Float, LAYOUT> p,
                             Float rho,
                             Float mu,
                             const Vec2<Float>& w,
                             Scalar<Float, LAYOUT> FUX,
                             VertexScalar<Float, LAYOUT> FUY,
                             VertexScalar<Float, LAYOUT> FVX,
                             Scalar<Float, LAYOUT> FVY) {
  return calc_mom_flux(grid, u, p, rho, mu, w, FUX, FUY, FVX, FVY, None{});
}

template <typename Float, Layout LAYOUT>
constexpr void calc_mom_flux(const Grid<Float, LAYOUT>& grid,
                             const FaceVector<Float, LAYOUT> u,
                             const Scalar<Float, LAYOUT> p,
                             Float rho,
                             Float mu,
                             Scalar<Float, LAYOUT> FUX,
                             VertexScalar<Float, LAYOUT> FUY,
                             VertexScalar<Float, LAYOUT> FVX,
                             Scalar<Float, LAYOUT> FVY,
                             Scalar<Float, LAYOUT> FWZ) {
  return calc_mom_flux(grid, u, p, rho, mu, None{}, FUX, FUY, FVX, FVY, FWZ);
}
// =================================================================================================
template <typename Float,
          Layout LAYOUT,
          IsNoneOr<Scalar<Float, LAYOUT>> J_t,
          IsNoneOr<Scalar<Float, LAYOUT>> FWZ_t>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const J_t J_old,
                        const J_t J,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const FWZ_t FWZ,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  dispatch_metric(
      grid, DISPATCH_FUNC {
        if constexpr (Metric::is_2d != IsNone<FWZ_t>) {
          Igor::Panic("`FWZ` must be given iff coordinates are quasi-3D");
        } else {
          OrthogonalCoordinates::update_u<Metric>(
              grid, dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, u_old, u);
        }
      });
}

template <typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  update_u(grid, dt, None{}, None{}, FUX, FUY, FVX, FVY, None{}, u_old, u);
}

template <typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> J_old,
                        const Scalar<Float, LAYOUT> J,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  update_u(grid, dt, J_old, J, FUX, FUY, FVX, FVY, None{}, u_old, u);
}

template <typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const Scalar<Float, LAYOUT> FWZ,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  update_u(grid, dt, None{}, None{}, FUX, FUY, FVX, FVY, FWZ, u_old, u);
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> J_old,
                        const Scalar<Float, LAYOUT> J,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const Scalar<Float, LAYOUT> FWZ,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN: Igor::Panic("Do not use `FWZ` for Cartesian coordinates.");
    case Coordinates::POLAR:     Igor::Panic("Do not use `FWZ` for polar coordinates.");
    case Coordinates::SYMMETRIC_SPHERICAL:
      return OrthogonalCoordinates::update_u<Metric::SymmetricSpherical>(
          grid, dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, u_old, u);
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
  dispatch_metric(
      grid,
      DISPATCH_FUNC { OrthogonalCoordinates::correct_velocity<Metric>(grid, dp, rho, dt, u, p); });
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
  const auto hx = grid.coords() == Coordinates::CARTESIAN ? grid.dx() : grid.ym(0) * grid.dx();
  const auto hy = grid.dy();

  // Advection: dt * (|u|/hx + |v|/hy) <= CFL
  const auto adv = ux_max / hx + uy_max / hy;
  // Diffusion: dt * 2 * nu * (1/hx^2 + 1/hy^2) <= CFL
  const auto diff         = 2.0 * (mu / rho) * (1.0 / Igor::sqr(hx) + 1.0 / Igor::sqr(hy));

  constexpr auto no_limit = std::numeric_limits<Float>::max();
  return std::min(adv > 0.0 ? CFL / adv : no_limit,  //
                  diff > 0.0 ? CFL / diff : no_limit);
}
