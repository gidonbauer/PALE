#pragma once

#include <Igor/Math.hpp>

#include "Grid.hpp"
#include "Metrics.hpp"
#include "WENO5.hpp"

namespace Orthogonal {

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   const FaceVector<Float, LAYOUT> sL,
                                   const FaceVector<Float, LAYOUT> sR,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto si     = u.x(i, j) >= 0.0 ? sL.x(i, j) : sR.x(i, j);
    const auto dsdx   = (s(i, j) - s(i - 1, j)) / grid.dx();
    const auto inv_h1 = 1.0 / Metric::h1(grid.x(i), grid.ym(j));
    F.x(i, j)         = si * u.x(i, j) - D * inv_h1 * dsdx;
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto si     = u.y(i, j) >= 0.0 ? sL.y(i, j) : sR.y(i, j);
    const auto dsdy   = (s(i, j) - s(i, j - 1)) / grid.dy();
    const auto inv_h2 = 1.0 / Metric::h2(grid.xm(i), grid.y(j));
    F.y(i, j)         = si * u.y(i, j) - D * inv_h2 * dsdy;
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   const FaceVector<Float, LAYOUT> sL,
                                   const FaceVector<Float, LAYOUT> sR,
                                   const Vec2<Float>& w,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto si     = (u.x(i, j) - w.x) >= 0.0 ? sL.x(i, j) : sR.x(i, j);
    const auto dsdx   = (s(i, j) - s(i - 1, j)) / grid.dx();
    const auto inv_h1 = 1.0 / Metric::h1(grid.x(i), grid.ym(j));
    F.x(i, j)         = si * (u.x(i, j) - w.x) - D * inv_h1 * dsdx;
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto si     = (u.y(i, j) - w.y) >= 0.0 ? sL.y(i, j) : sR.y(i, j);
    const auto dsdy   = (s(i, j) - s(i, j - 1)) / grid.dy();
    const auto inv_h2 = 1.0 / Metric::h2(grid.xm(i), grid.y(j));
    F.y(i, j)         = si * (u.y(i, j) - w.y) - D * inv_h2 * dsdy;
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const FaceVector<Float, LAYOUT> F,
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

    const auto inv_H = 1.0 / Metric::H(grid.xm(i), grid.ym(j));

    s(i, j)          = s_old(i, j) - dt * inv_H * (dFdq1 + dFdq2);
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> J_old,
                        const Scalar<Float, LAYOUT> J,
                        const FaceVector<Float, LAYOUT> F,
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

    // const auto H_old  = Metric::H(grid.xm(i) - Delta_old.x, grid.ym(j) - Delta_old.y);
    // const auto H_flux = Metric::H(grid.xm(i) - Delta_flux.x, grid.ym(j) - Delta_flux.y);
    const auto inv_H = 1.0 / Metric::H(grid.xm(i), grid.ym(j));

    // const auto div_w =
    //     inv_H *
    //     (w.x * (Metric::h2(grid.xm(i), grid.ym(j)) * Metric::dh3_dq1(grid.xm(i), grid.ym(j)) +
    //             Metric::h3(grid.xm(i), grid.ym(j)) * Metric::dh2_dq1(grid.xm(i), grid.ym(j))) +
    //      w.y * (Metric::h1(grid.xm(i), grid.ym(j)) * Metric::dh3_dq2(grid.xm(i), grid.ym(j)) +
    //             Metric::h3(grid.xm(i), grid.ym(j)) * Metric::dh1_dq2(grid.xm(i), grid.ym(j))));

    s(i, j) = (J_old(i, j) * s_old(i, j) - dt * inv_H * (dFdq1 + dFdq2)) / J(i, j);
    // s(i, j) = s_old(i, j) - dt * (inv_H * (dFdq1 + dFdq2) + s(i, j) * div_w);
    // s(i, j) = (H_old * s_old(i, j) - dt * (dFdq1 + dFdq2)) * inv_H;
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> src,
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

    const auto H = Metric::H(grid.xm(i), grid.ym(j));

    s(i, j)      = s_old(i, j) - dt * ((dFdq1 + dFdq2) / H - H * src(i, j));
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
template <typename Float, Layout LAYOUT>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  static auto sL = grid.alloc_face_vector();
  static auto sR = grid.alloc_face_vector();
  weno_reconstruction(grid, s, sL, sR);

  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return Orthogonal::calc_advection_flux<Metric::Cartesian>(grid, u, s, sL, sR, D, F);
    case Coordinates::POLAR:
      return Orthogonal::calc_advection_flux<Metric::Polar>(grid, u, s, sL, sR, D, F);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return Orthogonal::calc_advection_flux<Metric::SymmetricSpherical>(grid, u, s, sL, sR, D, F);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_advection_flux(const Grid<Float, LAYOUT>& grid,
                                   const FaceVector<Float, LAYOUT> u,
                                   const Scalar<Float, LAYOUT> s,
                                   const Vec2<Float>& w,
                                   Float D,
                                   FaceVector<Float, LAYOUT> F) {
  static auto sL = grid.alloc_face_vector();
  static auto sR = grid.alloc_face_vector();
  weno_reconstruction(grid, s, sL, sR);

  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return Orthogonal::calc_advection_flux<Metric::Cartesian>(grid, u, s, sL, sR, w, D, F);
    case Coordinates::POLAR:
      return Orthogonal::calc_advection_flux<Metric::Polar>(grid, u, s, sL, sR, w, D, F);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return Orthogonal::calc_advection_flux<Metric::SymmetricSpherical>(
          grid, u, s, sL, sR, w, D, F);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return Orthogonal::update_s<Metric::Cartesian>(grid, dt, F, s_old, s);
    case Coordinates::POLAR: return Orthogonal::update_s<Metric::Polar>(grid, dt, F, s_old, s);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return Orthogonal::update_s<Metric::SymmetricSpherical>(grid, dt, F, s_old, s);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> J_old,
                        const Scalar<Float, LAYOUT> J,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return Orthogonal::update_s<Metric::Cartesian>(grid, dt, J_old, J, F, s_old, s);
    case Coordinates::POLAR:
      return Orthogonal::update_s<Metric::Polar>(grid, dt, J_old, J, F, s_old, s);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return Orthogonal::update_s<Metric::SymmetricSpherical>(grid, dt, J_old, J, F, s_old, s);
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void update_s(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const FaceVector<Float, LAYOUT> F,
                        const Scalar<Float, LAYOUT> src,
                        const Scalar<Float, LAYOUT> s_old,
                        Scalar<Float, LAYOUT> s) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      return Orthogonal::update_s<Metric::Cartesian>(grid, dt, F, src, s_old, s);
    case Coordinates::POLAR: return Orthogonal::update_s<Metric::Polar>(grid, dt, F, src, s_old, s);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return Orthogonal::update_s<Metric::SymmetricSpherical>(grid, dt, F, src, s_old, s);
  }
  Igor::Panic("Unreachable");
}
