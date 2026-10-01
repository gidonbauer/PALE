#pragma once

#include <Igor/Math.hpp>

#include "Grid.hpp"
#include "WENO5.hpp"

namespace Orthogonal {

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT, IsNoneOr<Vec2<Float>> W_t>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   const FaceVector<Float, LAYOUT> sL,
                                   const FaceVector<Float, LAYOUT> sR,
                                   const W_t& w,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto wx     = IF_NONE_ELSE(W_t, 0.0, w.x);
    const auto si     = (u.x(i, j) - wx) >= 0.0 ? sL.x(i, j) : sR.x(i, j);
    const auto dsdx   = (s(i, j) - s(i - 1, j)) / grid.dx();
    const auto inv_h1 = 1.0 / Metric::h1(grid.x(i), grid.ym(j));
    F.x(i, j)         = si * (u.x(i, j) - wx) - D * inv_h1 * dsdx;
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto wy     = IF_NONE_ELSE(W_t, 0.0, w.y);
    const auto si     = (u.y(i, j) - wy) >= 0.0 ? sL.y(i, j) : sR.y(i, j);
    const auto dsdy   = (s(i, j) - s(i, j - 1)) / grid.dy();
    const auto inv_h2 = 1.0 / Metric::h2(grid.xm(i), grid.y(j));
    F.y(i, j)         = si * (u.y(i, j) - wy) - D * inv_h2 * dsdy;
  });
}

// =================================================================================================
template <typename Metric,
          typename Float,
          Layout LAYOUT,
          IsNoneOr<Scalar<Float, LAYOUT>> J_t,
          IsNoneOr<Scalar<Float, LAYOUT>> SRC_t>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const J_t J_old,
                        const J_t J,
                        const FaceVector<Float, LAYOUT> F,
                        const SRC_t src,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  grid.foreach_i(FOREACH_FUNC {
    const auto H_right  = Metric::H(grid.x(i + 1), grid.ym(j));
    const auto h1_right = Metric::h1(grid.x(i + 1), grid.ym(j));
    const auto H_left   = Metric::H(grid.x(i), grid.ym(j));
    const auto h1_left  = Metric::h1(grid.x(i), grid.ym(j));
    const auto dFdq1 =
        (H_right / h1_right * F.right(i, j) - H_left / h1_left * F.left(i, j)) / grid.dx();

    const auto H_top     = Metric::H(grid.xm(i), grid.y(j + 1));
    const auto h2_top    = Metric::h2(grid.xm(i), grid.y(j + 1));
    const auto H_bottom  = Metric::H(grid.xm(i), grid.y(j));
    const auto h2_bottom = Metric::h2(grid.xm(i), grid.y(j));
    const auto dFdq2 =
        (H_top / h2_top * F.top(i, j) - H_bottom / h2_bottom * F.bottom(i, j)) / grid.dy();

    const auto H          = Metric::H(grid.xm(i), grid.ym(j));
    const auto Ji_old     = IF_NONE_ELSE(J_t, 1.0, J_old(i, j));
    const auto Ji         = IF_NONE_ELSE(J_t, 1.0, J(i, j));
    const auto div_factor = IF_NONE_ELSE(J_t, 1.0 / H, 1.0);

    const auto srci       = IF_NONE_ELSE(SRC_t, 0.0, src(i, j));

    s(i, j) = (Ji_old * s_old(i, j) - dt * div_factor * (dFdq1 + dFdq2) + dt * H * srci) / Ji;
  });
}
}  // namespace Orthogonal

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr auto advection_adjust_dt(const Grid<Float, LAYOUT>& grid, Float D, Float CFL) -> Float {
  IGOR_ASSERT(D >= 0.0, "Diffusion coefficient cannot be negative but is {}", D);
  const auto h = std::min(grid.dx(), grid.dy());
  return D > 0.0 ? CFL * 0.25 * Igor::sqr(h) / D : std::numeric_limits<Float>::max();
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Vec2<Float>> W_t>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   const W_t& w,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  static auto sL = grid.alloc_face_vector();
  static auto sR = grid.alloc_face_vector();
  weno_reconstruction(grid, s, sL, sR);
  dispatch_metric(
      grid,
      DISPATCH_FUNC { Orthogonal::calc_advection_flux<Metric>(grid, u, s, sL, sR, w, D, F); });
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  calc_advection_flux(grid, u, s, None{}, D, F);
}

// =================================================================================================
template <typename Float,
          Layout LAYOUT,
          IsNoneOr<Scalar<Float, LAYOUT>> J_t,
          IsNoneOr<Scalar<Float, LAYOUT>> SRC_t>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const J_t J_old,
                        const J_t J,
                        const FaceVector<Float, LAYOUT> F,
                        const SRC_t src,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  dispatch_metric(
      grid, DISPATCH_FUNC { Orthogonal::update_s<Metric>(grid, dt, J_old, J, F, src, s_old, s); });
}

template <typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  return update_s(grid, dt, None{}, None{}, F, None{}, s_old, s);
}

template <typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> J_old,
                        const Scalar<Float, LAYOUT> J,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  return update_s(grid, dt, J_old, J, F, None{}, s_old, s);
}

template <typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> src,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  return update_s(grid, dt, None{}, None{}, F, src, s_old, s);
}
