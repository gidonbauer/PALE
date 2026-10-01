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
// We assume that the ALE mesh movement is only in the axis directions, meaning only in x-/theta- or
// y-/r- direction. Additionally, we assume that the grid velocity is the same everywhere. This
// means that the Jacobian matrix is the identity matrix and its determinant is only dependend on
// the coordinate system. The flux is then almost identical to the non-ALE case, the only difference
// are that the radius `r` changes, this is done in the `Grid` class, and the addition of the grid
// veclocity `w`. The fluxes `FUY` and `FVX` are then no longer the same.
template <typename Metric,
          typename Float,
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
  static_assert((Metric::is_2d && IsNone<FWZ_t>) || (!Metric::is_2d && !IsNone<FWZ_t>),
                "Provide `Scalar` for `FWZ` in quasi-3D case and `None` in 2D case.");

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
    if constexpr (!IsNone<W_t>) { FUX(i, j) -= u1 * w.x; }

    FVY(i, j) =
        Igor::sqr(u2) + p(i, j) / rho -
        2.0 * nu * (du2dq2 * inv_h2 + u1 * inv_h1h2 * Metric::dh2_dq1(grid.xm(i), grid.ym(j)));
    if constexpr (!IsNone<W_t>) { FVY(i, j) -= u2 * w.y; }

    if constexpr (!IsNone<FWZ_t>) {
      FWZ(i, j) =
          p(i, j) / rho - 2.0 * nu *
                              (u1 * inv_h1 * inv_h3 * Metric::dh3_dq1(grid.xm(i), grid.ym(j)) +
                               u2 * inv_h2 * inv_h3 * Metric::dh3_dq2(grid.xm(i), grid.ym(j)));
    }
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
    if constexpr (!IsNone<W_t>) { FUY(i, j) -= u1 * w.y; }

    FVX(i, j) = u1 * u2 - nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));
    if constexpr (!IsNone<W_t>) { FVX(i, j) -= u2 * w.x; }
  });
}

// =================================================================================================
template <typename Metric,
          typename Float,
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
  static_assert((Metric::is_2d && IsNone<FWZ_t>) || (!Metric::is_2d && !IsNone<FWZ_t>),
                "Provide `Scalar` for `FWZ` in quasi-3D case and `None` in 2D case.");

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

    const auto T12        = (FVX(i, j + 1) + FVX(i, j)) / 2.0;
    const auto T22        = (FVY(i, j) + FVY(i - 1, j)) / 2.0;
    const auto T33        = If_NONE_ELSE(FWZ_t, 0.0, (FWZ(i, j) + FWZ(i - 1, j)) / 2.0);

    const auto h2         = Metric::h2(grid.x(i), grid.ym(j));
    const auto h3         = Metric::h3(grid.x(i), grid.ym(j));
    const auto inv_H      = 1.0 / Metric::H(grid.x(i), grid.ym(j));

    const auto Ji_old     = If_NONE_ELSE(J_t, 1.0, (J_old(i, j) + J_old(i - 1, j)) / 2.0);
    const auto Ji         = If_NONE_ELSE(J_t, 1.0, (J(i, j) + J(i - 1, j)) / 2.0);
    const auto div_factor = If_NONE_ELSE(J_t, inv_H, 1.0);

    u.x(i, j) = (Ji_old * u_old.x(i, j) - dt * div_factor *
                                              (dHT11_dq1 + dHT21_dq2 +  //
                                               h3 * T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) -
                                               h3 * T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j)) -
                                               h2 * T33 * Metric::dh3_dq1(grid.x(i), grid.ym(j)))) /
                Ji;
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

    const auto T11        = (FUX(i, j) + FUX(i, j - 1)) / 2.0;
    const auto T21        = (FVX(i + 1, j) + FVX(i, j)) / 2.0;
    const auto T33        = If_NONE_ELSE(FWZ_t, 0.0, (FWZ(i, j) + FWZ(i, j - 1)) / 2.0);

    const auto h1         = Metric::h1(grid.xm(i), grid.y(j));
    const auto h3         = Metric::h3(grid.xm(i), grid.y(j));
    const auto inv_H      = 1.0 / Metric::H(grid.xm(i), grid.y(j));

    const auto Ji_old     = If_NONE_ELSE(J_t, 1.0, (J_old(i, j) + J_old(i, j - 1)) / 2.0);
    const auto Ji         = If_NONE_ELSE(J_t, 1.0, (J(i, j) + J(i, j - 1)) / 2.0);
    const auto div_factor = If_NONE_ELSE(J_t, inv_H, 1.0);

    u.y(i, j) = (Ji_old * u_old.y(i, j) - dt * div_factor *
                                              (dHT12_dq1 + dHT22_dq2 +  //
                                               h3 * T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -
                                               h3 * T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j)) -
                                               h1 * T33 * Metric::dh3_dq2(grid.xm(i), grid.y(j)))) /
                Ji;
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
