#pragma once

#include <Igor/Math.hpp>

#include "Grid.hpp"

namespace OrthogonalCoordinates {

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_div(const Grid<Float, LAYOUT>& grid,
                        const FaceVector<Float, LAYOUT> uf,
                        Scalar<Float, LAYOUT> div) {
  grid.foreach_i(FOREACH_FUNC {
    const auto H_left   = Metric::H(grid.x(i), grid.ym(j));
    const auto h1_left  = Metric::h1(grid.x(i), grid.ym(j));
    const auto H_right  = Metric::H(grid.x(i + 1), grid.ym(j));
    const auto h1_right = Metric::h1(grid.x(i + 1), grid.ym(j));
    const auto dq1 =
        (H_right / h1_right * uf.right(i, j) - H_left / h1_left * uf.left(i, j)) / grid.dx();

    const auto H_bottom  = Metric::H(grid.xm(i), grid.y(j));
    const auto h2_bottom = Metric::h2(grid.xm(i), grid.y(j));
    const auto H_top     = Metric::H(grid.xm(i), grid.y(j + 1));
    const auto h2_top    = Metric::h2(grid.xm(i), grid.y(j + 1));
    const auto dq2 =
        (H_top / h2_top * uf.top(i, j) - H_bottom / h2_bottom * uf.bottom(i, j)) / grid.dy();

    const auto H_mid = Metric::H(grid.thetam(i), grid.rm(j));
    div(i, j)        = (dq1 + dq2) / H_mid;
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_mom_flux(const Grid<Float, LAYOUT>& grid,
                             const FaceVector<Float, LAYOUT> u,
                             const Scalar<Float, LAYOUT> p,
                             Float rho,
                             Float mu,
                             Scalar<Float, LAYOUT> FUX,
                             VertexScalar<Float, LAYOUT> FUY,
                             VertexScalar<Float, LAYOUT> FVX,
                             Scalar<Float, LAYOUT> FVY) {
  static_assert(Metric::is_2d,
                "Call the `calc_mom_flux` overload with `FWZ` for 3D symmetric cases.");

  const auto nu = mu / rho;
  grid.foreach_a(FOREACH_FUNC {
    const auto u1       = (u.right(i, j) + u.left(i, j)) / 2.0;
    const auto u2       = (u.top(i, j) + u.bottom(i, j)) / 2.0;
    const auto du1dq1   = (u.right(i, j) - u.left(i, j)) / grid.dx();
    const auto du2dq2   = (u.top(i, j) - u.bottom(i, j)) / grid.dy();

    const auto inv_h1   = 1.0 / Metric::h1(grid.xm(i), grid.ym(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.xm(i), grid.ym(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUX(i, j) =
        Igor::sqr(u1) + p(i, j) / rho -
        2.0 * nu * (du1dq1 * inv_h1 + u2 * inv_h1h2 * Metric::dh1_dq2(grid.xm(i), grid.ym(j)));

    FVY(i, j) =
        Igor::sqr(u2) + p(i, j) / rho -
        2.0 * nu * (du2dq2 * inv_h2 + u1 * inv_h1h2 * Metric::dh2_dq1(grid.xm(i), grid.ym(j)));
  });

  grid.foreach_vertex_i(FOREACH_FUNC {
    const auto u1       = (u.x(i, j) + u.x(i, j - 1)) / 2.0;
    const auto u2       = (u.y(i, j) + u.y(i - 1, j)) / 2.0;
    const auto du1dq2   = (u.x(i, j) - u.x(i, j - 1)) / grid.dy();
    const auto du2dq1   = (u.y(i, j) - u.y(i - 1, j)) / grid.dx();

    const auto inv_h1   = 1.0 / Metric::h1(grid.x(i), grid.y(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.x(i), grid.y(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUY(i, j)           = u1 * u2 - nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                          u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                          u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));

    FVX(i, j)           = u1 * u2 - nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                          u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                          u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));
  });
}

// =================================================================================================
// We assume that the ALE mesh movement is only in the axis directions, meaning only in x-/theta- or
// y-/r- direction. Additionally, we assmue that the grid velocity is the same everywhere. This
// means that the Jacobian matrix is the identity matrix and its determinant is one. The flux then
// almost identical to the non-ALE case, the only difference are that the radius `r` changes, this
// is done in the `Grid` class, and the addition of the grid veclocity `w`. The fluxes `FUY` and
// `FVX` are then no longer the same.
template <typename Metric, typename Float, Layout LAYOUT>
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
  static_assert(Metric::is_2d,
                "Call the `calc_mom_flux` overload with `FWZ` for 3D symmetric cases.");

  const auto nu = mu / rho;
  grid.foreach_a(FOREACH_FUNC {
    const auto u1       = (u.right(i, j) + u.left(i, j)) / 2.0;
    const auto u2       = (u.top(i, j) + u.bottom(i, j)) / 2.0;
    const auto du1dq1   = (u.right(i, j) - u.left(i, j)) / grid.dx();
    const auto du2dq2   = (u.top(i, j) - u.bottom(i, j)) / grid.dy();

    const auto inv_h1   = 1.0 / Metric::h1(grid.xm(i), grid.ym(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.xm(i), grid.ym(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUX(i, j) =
        Igor::sqr(u1) + p(i, j) / rho -
        2.0 * nu * (du1dq1 * inv_h1 + u2 * inv_h1h2 * Metric::dh1_dq2(grid.xm(i), grid.ym(j))) -
        u1 * w.x;

    FVY(i, j) =
        Igor::sqr(u2) + p(i, j) / rho -
        2.0 * nu * (du2dq2 * inv_h2 + u1 * inv_h1h2 * Metric::dh2_dq1(grid.xm(i), grid.ym(j))) -
        u2 * w.y;
  });

  grid.foreach_vertex_i(FOREACH_FUNC {
    const auto u1       = (u.x(i, j) + u.x(i, j - 1)) / 2.0;
    const auto u2       = (u.y(i, j) + u.y(i - 1, j)) / 2.0;
    const auto du1dq2   = (u.x(i, j) - u.x(i, j - 1)) / grid.dy();
    const auto du2dq1   = (u.y(i, j) - u.y(i - 1, j)) / grid.dx();

    const auto inv_h1   = 1.0 / Metric::h1(grid.x(i), grid.y(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.x(i), grid.y(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUY(i, j)           = u1 * u2 -
                          nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j))) -
                          u1 * w.y;

    FVX(i, j)           = u1 * u2 -
                          nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j))) -
                          u2 * w.x;
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
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
  static_assert(!Metric::is_2d,
                "Call the `calc_mom_flux` overload without `FWZ` for true 2D cases.");

  const auto nu = mu / rho;
  grid.foreach_a(FOREACH_FUNC {
    const auto u1       = (u.right(i, j) + u.left(i, j)) / 2.0;
    const auto u2       = (u.top(i, j) + u.bottom(i, j)) / 2.0;
    const auto du1dq1   = (u.right(i, j) - u.left(i, j)) / grid.dx();
    const auto du2dq2   = (u.top(i, j) - u.bottom(i, j)) / grid.dy();

    const auto inv_h1   = 1.0 / Metric::h1(grid.xm(i), grid.ym(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.xm(i), grid.ym(j));
    const auto inv_h3   = 1.0 / Metric::h3(grid.xm(i), grid.ym(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUX(i, j) =
        Igor::sqr(u1) + p(i, j) / rho -
        2.0 * nu * (du1dq1 * inv_h1 + u2 * inv_h1h2 * Metric::dh1_dq2(grid.xm(i), grid.ym(j)));

    FVY(i, j) =
        Igor::sqr(u2) + p(i, j) / rho -
        2.0 * nu * (du2dq2 * inv_h2 + u1 * inv_h1h2 * Metric::dh2_dq1(grid.xm(i), grid.ym(j)));

    FWZ(i, j) =
        p(i, j) / rho - 2.0 * nu *
                            (u1 * inv_h1 * inv_h3 * Metric::dh3_dq1(grid.xm(i), grid.ym(j)) +
                             u2 * inv_h2 * inv_h3 * Metric::dh3_dq2(grid.xm(i), grid.ym(j)));
  });

  grid.foreach_vertex_i(FOREACH_FUNC {
    const auto u1       = (u.x(i, j) + u.x(i, j - 1)) / 2.0;
    const auto u2       = (u.y(i, j) + u.y(i - 1, j)) / 2.0;
    const auto du1dq2   = (u.x(i, j) - u.x(i, j - 1)) / grid.dy();
    const auto du2dq1   = (u.y(i, j) - u.y(i - 1, j)) / grid.dx();

    const auto inv_h1   = 1.0 / Metric::h1(grid.x(i), grid.y(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.x(i), grid.y(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUY(i, j)           = u1 * u2 - nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                          u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                          u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));

    FVX(i, j)           = u1 * u2 - nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                          u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                          u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));
  });
}

// =================================================================================================
// We assume that the ALE mesh movement is only in the axis directions, meaning only in x-/theta- or
// y-/r- direction. Additionally, we assmue that the grid velocity is the same everywhere. This
// means that the Jacobian matrix is the identity matrix and its determinant is one. The flux then
// almost identical to the non-ALE case, the only difference are that the radius `r` changes, this
// is done in the `Grid` class, and the addition of the grid veclocity `w`. The fluxes `FUY` and
// `FVX` are then no longer the same.
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_mom_flux(const Grid<Float, LAYOUT>& grid,
                             const FaceVector<Float, LAYOUT> u,
                             const Scalar<Float, LAYOUT> p,
                             Float rho,
                             Float mu,
                             const Vec2<Float>& w,
                             Scalar<Float, LAYOUT> FUX,
                             VertexScalar<Float, LAYOUT> FUY,
                             VertexScalar<Float, LAYOUT> FVX,
                             Scalar<Float, LAYOUT> FVY,
                             Scalar<Float, LAYOUT> FWZ) {
  static_assert(!Metric::is_2d,
                "Call the `calc_mom_flux` overload without `FWZ` for true 2D cases.");

  const auto nu = mu / rho;
  grid.foreach_a(FOREACH_FUNC {
    const auto u1       = (u.right(i, j) + u.left(i, j)) / 2.0;
    const auto u2       = (u.top(i, j) + u.bottom(i, j)) / 2.0;
    const auto du1dq1   = (u.right(i, j) - u.left(i, j)) / grid.dx();
    const auto du2dq2   = (u.top(i, j) - u.bottom(i, j)) / grid.dy();

    const auto inv_h1   = 1.0 / Metric::h1(grid.xm(i), grid.ym(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.xm(i), grid.ym(j));
    const auto inv_h3   = 1.0 / Metric::h3(grid.xm(i), grid.ym(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUX(i, j) =
        Igor::sqr(u1) + p(i, j) / rho -
        2.0 * nu * (du1dq1 * inv_h1 + u2 * inv_h1h2 * Metric::dh1_dq2(grid.xm(i), grid.ym(j))) -
        u1 * w.x;

    FVY(i, j) =
        Igor::sqr(u2) + p(i, j) / rho -
        2.0 * nu * (du2dq2 * inv_h2 + u1 * inv_h1h2 * Metric::dh2_dq1(grid.xm(i), grid.ym(j))) -
        u2 * w.y;

    FWZ(i, j) =
        p(i, j) / rho - 2.0 * nu *
                            (u1 * inv_h1 * inv_h3 * Metric::dh3_dq1(grid.xm(i), grid.ym(j)) +
                             u2 * inv_h2 * inv_h3 * Metric::dh3_dq2(grid.xm(i), grid.ym(j)));
  });

  grid.foreach_vertex_i(FOREACH_FUNC {
    const auto u1       = (u.x(i, j) + u.x(i, j - 1)) / 2.0;
    const auto u2       = (u.y(i, j) + u.y(i - 1, j)) / 2.0;
    const auto du1dq2   = (u.x(i, j) - u.x(i, j - 1)) / grid.dy();
    const auto du2dq1   = (u.y(i, j) - u.y(i - 1, j)) / grid.dx();

    const auto inv_h1   = 1.0 / Metric::h1(grid.x(i), grid.y(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.x(i), grid.y(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    FUY(i, j)           = u1 * u2 -
                          nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j))) -
                          u1 * w.y;

    FVX(i, j)           = u1 * u2 -
                          nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j))) -
                          u2 * w.x;
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  static_assert(Metric::is_2d, "Call the `update_u` overload with `FWZ` for 3D symmetric cases.");
  // T11 = FUX
  // T12 = FVX
  // T21 = FUY
  // T22 = FVY

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto dHT11_dq1 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h1(grid.xm(i), grid.ym(j)) * FUX(i, j) -
         Metric::H(grid.xm(i - 1), grid.ym(j)) / Metric::h1(grid.xm(i - 1), grid.ym(j)) *
             FUX(i - 1, j)) /
        grid.dx();

    const auto dHT21_dq2 =
        (Metric::H(grid.x(i), grid.y(j + 1)) / Metric::h2(grid.x(i), grid.y(j + 1)) *
             FUY(i, j + 1) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h2(grid.x(i), grid.y(j)) * FUY(i, j)) /
        grid.dy();

    const auto T12   = (FVX(i, j + 1) + FVX(i, j)) / 2.0;
    const auto T22   = (FVY(i, j) + FVY(i - 1, j)) / 2.0;

    const auto inv_H = 1.0 / Metric::H(grid.x(i), grid.ym(j));
    const auto inv_h1h2 =
        1.0 / (Metric::h1(grid.x(i), grid.ym(j)) * Metric::h2(grid.x(i), grid.ym(j)));

    u.x(i, j) = u_old.x(i, j) - dt * (inv_H * (dHT11_dq1 + dHT21_dq2) +
                                      inv_h1h2 * (T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) +
                                                  T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j))));
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto dHT12_dq1 =
        (Metric::H(grid.x(i + 1), grid.y(j)) / Metric::h1(grid.x(i + 1), grid.y(j)) *
             FVX(i + 1, j) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h1(grid.x(i), grid.y(j)) * FVX(i, j)) /
        grid.dx();

    const auto dHT22_dq2 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h2(grid.xm(i), grid.ym(j)) * FVY(i, j) -
         Metric::H(grid.xm(i), grid.ym(j - 1)) / Metric::h2(grid.xm(i), grid.ym(j - 1)) *
             FVY(i, j - 1)) /
        grid.dy();

    const auto T21   = (FVX(i + 1, j) + FVX(i, j)) / 2.0;
    const auto T11   = (FUX(i, j) + FUX(i, j - 1)) / 2.0;

    const auto inv_H = 1.0 / Metric::H(grid.xm(i), grid.y(j));
    const auto inv_h1h2 =
        1.0 / (Metric::h1(grid.xm(i), grid.y(j)) * Metric::h2(grid.xm(i), grid.y(j)));

    u.y(i, j) = u_old.y(i, j) - dt * (inv_H * (dHT12_dq1 + dHT22_dq2) +
                                      inv_h1h2 * (T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -
                                                  T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j))));
  });
}

// =================================================================================================
// Here we need to account for the increase of cell size, this is done through the additional term
// $\vec{u} (\nabla \cdot \vec{w})$.
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Vec2<Float>& w,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  static_assert(Metric::is_2d, "Call the `update_u` overload with `FWZ` for 3D symmetric cases.");
  // T11 = FUX
  // T12 = FVX
  // T21 = FUY
  // T22 = FVY

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto dHT11_dq1 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h1(grid.xm(i), grid.ym(j)) * FUX(i, j) -
         Metric::H(grid.xm(i - 1), grid.ym(j)) / Metric::h1(grid.xm(i - 1), grid.ym(j)) *
             FUX(i - 1, j)) /
        grid.dx();

    const auto dHT21_dq2 =
        (Metric::H(grid.x(i), grid.y(j + 1)) / Metric::h2(grid.x(i), grid.y(j + 1)) *
             FUY(i, j + 1) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h2(grid.x(i), grid.y(j)) * FUY(i, j)) /
        grid.dy();

    const auto T12      = (FVX(i, j + 1) + FVX(i, j)) / 2.0;
    const auto T22      = (FVY(i, j) + FVY(i - 1, j)) / 2.0;

    const auto inv_H    = 1.0 / Metric::H(grid.x(i), grid.ym(j));
    const auto h1       = Metric::h1(grid.x(i), grid.ym(j));
    const auto h2       = Metric::h2(grid.x(i), grid.ym(j));
    const auto h3       = Metric::h3(grid.x(i), grid.ym(j));
    const auto inv_h1h2 = 1.0 / (h1 * h2);

    const auto div_w    = inv_H * (w.x * (h2 * Metric::dh3_dq1(grid.x(i), grid.ym(j)) +
                                          h3 * Metric::dh2_dq1(grid.x(i), grid.ym(j))) +
                                   w.y * (h1 * Metric::dh3_dq2(grid.x(i), grid.ym(j)) +
                                          h3 * Metric::dh1_dq2(grid.x(i), grid.ym(j))));

    u.x(i, j) = u_old.x(i, j) - dt * (inv_H * (dHT11_dq1 + dHT21_dq2) +
                                      inv_h1h2 * (T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) +
                                                  T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j))) +
                                      u.x(i, j) * div_w);
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto dHT12_dq1 =
        (Metric::H(grid.x(i + 1), grid.y(j)) / Metric::h1(grid.x(i + 1), grid.y(j)) *
             FVX(i + 1, j) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h1(grid.x(i), grid.y(j)) * FVX(i, j)) /
        grid.dx();

    const auto dHT22_dq2 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h2(grid.xm(i), grid.ym(j)) * FVY(i, j) -
         Metric::H(grid.xm(i), grid.ym(j - 1)) / Metric::h2(grid.xm(i), grid.ym(j - 1)) *
             FVY(i, j - 1)) /
        grid.dy();

    const auto T21      = (FVX(i + 1, j) + FVX(i, j)) / 2.0;
    const auto T11      = (FUX(i, j) + FUX(i, j - 1)) / 2.0;

    const auto inv_H    = 1.0 / Metric::H(grid.xm(i), grid.y(j));
    const auto h1       = Metric::h1(grid.xm(i), grid.y(j));
    const auto h2       = Metric::h2(grid.xm(i), grid.y(j));
    const auto h3       = Metric::h3(grid.xm(i), grid.y(j));
    const auto inv_h1h2 = 1.0 / (h1 * h2);

    const auto div_w    = inv_H * (w.x * (h2 * Metric::dh3_dq1(grid.xm(i), grid.y(j)) +
                                          h3 * Metric::dh2_dq1(grid.xm(i), grid.y(j))) +
                                   w.y * (h1 * Metric::dh3_dq2(grid.xm(i), grid.y(j)) +
                                          h3 * Metric::dh1_dq2(grid.xm(i), grid.y(j))));

    u.y(i, j) = u_old.y(i, j) - dt * (inv_H * (dHT12_dq1 + dHT22_dq2) +
                                      inv_h1h2 * (T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -
                                                  T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j))) +
                                      u.y(i, j) * div_w);
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const Scalar<Float, LAYOUT> FWZ,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  static_assert(!Metric::is_2d, "Call the `update_u` overload without `FWZ` for true 2D cases.");

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto dHT11_dq1 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h1(grid.xm(i), grid.ym(j)) * FUX(i, j) -
         Metric::H(grid.xm(i - 1), grid.ym(j)) / Metric::h1(grid.xm(i - 1), grid.ym(j)) *
             FUX(i - 1, j)) /
        grid.dx();

    const auto dHT21_dq2 =
        (Metric::H(grid.x(i), grid.y(j + 1)) / Metric::h2(grid.x(i), grid.y(j + 1)) *
             FUY(i, j + 1) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h2(grid.x(i), grid.y(j)) * FUY(i, j)) /
        grid.dy();

    const auto T12      = (FVX(i, j + 1) + FVX(i, j)) / 2.0;
    const auto T22      = (FVY(i, j) + FVY(i - 1, j)) / 2.0;
    const auto T33      = (FWZ(i, j) + FWZ(i - 1, j)) / 2.0;

    const auto inv_H    = 1.0 / Metric::H(grid.x(i), grid.ym(j));
    const auto inv_h1   = 1.0 / Metric::h1(grid.x(i), grid.ym(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.x(i), grid.ym(j));
    const auto inv_h3   = 1.0 / Metric::h3(grid.x(i), grid.ym(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    u.x(i, j) =
        u_old.x(i, j) - dt * (inv_H * (dHT11_dq1 + dHT21_dq2) +
                              inv_h1h2 * (T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) +
                                          T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j))) -
                              inv_h1 * inv_h3 * T33 * Metric::dh3_dq1(grid.x(i), grid.ym(j)));
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto dHT12_dq1 =
        (Metric::H(grid.x(i + 1), grid.y(j)) / Metric::h1(grid.x(i + 1), grid.y(j)) *
             FVX(i + 1, j) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h1(grid.x(i), grid.y(j)) * FVX(i, j)) /
        grid.dx();

    const auto dHT22_dq2 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h2(grid.xm(i), grid.ym(j)) * FVY(i, j) -
         Metric::H(grid.xm(i), grid.ym(j - 1)) / Metric::h2(grid.xm(i), grid.ym(j - 1)) *
             FVY(i, j - 1)) /
        grid.dy();

    const auto T11      = (FUX(i, j) + FUX(i, j - 1)) / 2.0;
    const auto T21      = (FVX(i + 1, j) + FVX(i, j)) / 2.0;
    const auto T33      = (FWZ(i, j) + FWZ(i, j - 1)) / 2.0;

    const auto inv_H    = 1.0 / Metric::H(grid.xm(i), grid.y(j));
    const auto inv_h1   = 1.0 / Metric::h1(grid.xm(i), grid.y(j));
    const auto inv_h2   = 1.0 / Metric::h2(grid.xm(i), grid.y(j));
    const auto inv_h3   = 1.0 / Metric::h3(grid.xm(i), grid.y(j));
    const auto inv_h1h2 = inv_h1 * inv_h2;

    u.y(i, j) =
        u_old.y(i, j) - dt * (inv_H * (dHT12_dq1 + dHT22_dq2) +
                              inv_h1h2 * (T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -
                                          T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j))) -
                              inv_h2 * inv_h3 * T33 * Metric::dh3_dq2(grid.xm(i), grid.y(j)));
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_u(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Vec2<Float>& w,
                        const Scalar<Float, LAYOUT> FUX,
                        const VertexScalar<Float, LAYOUT> FUY,
                        const VertexScalar<Float, LAYOUT> FVX,
                        const Scalar<Float, LAYOUT> FVY,
                        const Scalar<Float, LAYOUT> FWZ,
                        const FaceVector<Float, LAYOUT> u_old,
                        FaceVector<Float, LAYOUT> u) {
  static_assert(!Metric::is_2d, "Call the `update_u` overload without `FWZ` for true 2D cases.");

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto dHT11_dq1 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h1(grid.xm(i), grid.ym(j)) * FUX(i, j) -
         Metric::H(grid.xm(i - 1), grid.ym(j)) / Metric::h1(grid.xm(i - 1), grid.ym(j)) *
             FUX(i - 1, j)) /
        grid.dx();

    const auto dHT21_dq2 =
        (Metric::H(grid.x(i), grid.y(j + 1)) / Metric::h2(grid.x(i), grid.y(j + 1)) *
             FUY(i, j + 1) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h2(grid.x(i), grid.y(j)) * FUY(i, j)) /
        grid.dy();

    const auto T12   = (FVX(i, j + 1) + FVX(i, j)) / 2.0;
    const auto T22   = (FVY(i, j) + FVY(i - 1, j)) / 2.0;
    const auto T33   = (FWZ(i, j) + FWZ(i - 1, j)) / 2.0;

    const auto inv_H = 1.0 / Metric::H(grid.x(i), grid.ym(j));
    const auto h1    = Metric::h1(grid.x(i), grid.ym(j));
    const auto h2    = Metric::h2(grid.x(i), grid.ym(j));
    const auto h3    = Metric::h3(grid.x(i), grid.ym(j));

    const auto div_w = inv_H * (w.x * (h2 * Metric::dh3_dq1(grid.x(i), grid.ym(j)) +
                                       h3 * Metric::dh2_dq1(grid.x(i), grid.ym(j))) +
                                w.y * (h1 * Metric::dh3_dq2(grid.x(i), grid.ym(j)) +
                                       h3 * Metric::dh1_dq2(grid.x(i), grid.ym(j))));

    u.x(i, j) = u_old.x(i, j) -
                dt * (inv_H * (dHT11_dq1 + dHT21_dq2) +
                      (T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) +
                       T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j))) /
                          (h1 * h2) -
                      T33 / (h1 * h3) * Metric::dh3_dq1(grid.x(i), grid.ym(j)) + u.x(i, j) * div_w);
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto dHT12_dq1 =
        (Metric::H(grid.x(i + 1), grid.y(j)) / Metric::h1(grid.x(i + 1), grid.y(j)) *
             FVX(i + 1, j) -
         Metric::H(grid.x(i), grid.y(j)) / Metric::h1(grid.x(i), grid.y(j)) * FVX(i, j)) /
        grid.dx();

    const auto dHT22_dq2 =
        (Metric::H(grid.xm(i), grid.ym(j)) / Metric::h2(grid.xm(i), grid.ym(j)) * FVY(i, j) -
         Metric::H(grid.xm(i), grid.ym(j - 1)) / Metric::h2(grid.xm(i), grid.ym(j - 1)) *
             FVY(i, j - 1)) /
        grid.dy();

    const auto T11   = (FUX(i, j) + FUX(i, j - 1)) / 2.0;
    const auto T21   = (FVX(i + 1, j) + FVX(i, j)) / 2.0;
    const auto T33   = (FWZ(i, j) + FWZ(i, j - 1)) / 2.0;

    const auto inv_H = 1.0 / Metric::H(grid.xm(i), grid.y(j));
    const auto h1    = Metric::h1(grid.xm(i), grid.y(j));
    const auto h2    = Metric::h2(grid.xm(i), grid.y(j));
    const auto h3    = Metric::h3(grid.xm(i), grid.y(j));

    const auto div_w = inv_H * (w.x * (h2 * Metric::dh3_dq1(grid.xm(i), grid.y(j)) +
                                       h3 * Metric::dh2_dq1(grid.xm(i), grid.y(j))) +
                                w.y * (h1 * Metric::dh3_dq2(grid.xm(i), grid.y(j)) +
                                       h3 * Metric::dh1_dq2(grid.xm(i), grid.y(j))));

    u.y(i, j) = u_old.y(i, j) -
                dt * (inv_H * (dHT12_dq1 + dHT22_dq2) +
                      (T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -
                       T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j))) /
                          (h1 * h2) -
                      T33 / (h2 * h3) * Metric::dh3_dq2(grid.xm(i), grid.y(j)) + u.y(i, j) * div_w);
  });
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void correct_velocity(const Grid<Float, LAYOUT>& grid,
                                const Scalar<Float, LAYOUT> dp,
                                Float rho,
                                Float dt,
                                FaceVector<Float, LAYOUT> u,
                                Scalar<Float, LAYOUT> p) {
  grid.foreach_a(FOREACH_FUNC { p(i, j) += dp(i, j); });

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto dpdq1   = (dp(i, j) - dp(i - 1, j)) / grid.dx();
    const auto inv_h1  = 1.0 / Metric::h1(grid.x(i), grid.ym(j));
    u.x(i, j)         -= (dt / rho) * inv_h1 * dpdq1;
  });
  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto dpdq2   = (dp(i, j) - dp(i, j - 1)) / grid.dy();
    const auto inv_h2  = 1.0 / Metric::h2(grid.xm(i), grid.y(j));
    u.y(i, j)         -= (dt / rho) * inv_h2 * dpdq2;
  });
}

}  // namespace OrthogonalCoordinates
