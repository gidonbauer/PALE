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
    const auto h2_left   = Metric::h2(grid.x(i), grid.ym(j));
    const auto h2_right  = Metric::h2(grid.x(i + 1), grid.ym(j));
    const auto dq1       = (h2_right * uf.right(i, j) - h2_left * uf.left(i, j)) / grid.dx();

    const auto h1_bottom = Metric::h1(grid.xm(i), grid.y(j));
    const auto h1_top    = Metric::h1(grid.xm(i), grid.y(j + 1));
    const auto dq2       = (h1_top * uf.top(i, j) - h1_bottom * uf.bottom(i, j)) / grid.dy();

    const auto h1_mid    = Metric::h1(grid.thetam(i), grid.rm(j));
    const auto h2_mid    = Metric::h2(grid.thetam(i), grid.rm(j));
    div(i, j)            = (dq1 + dq2) / (h1_mid * h2_mid);
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
        -Igor::sqr(u1) - p(i, j) / rho +
        2.0 * nu * (du1dq1 * inv_h1 + u2 * inv_h1h2 * Metric::dh1_dq2(grid.xm(i), grid.ym(j)));

    FVY(i, j) =
        -Igor::sqr(u2) - p(i, j) / rho +
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

    FUY(i, j)           = -u1 * u2 + nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                           u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                           u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));

    FVX(i, j)           = -u1 * u2 + nu * (du2dq1 * inv_h1 + du1dq2 * inv_h2 -
                                           u1 * inv_h1h2 * Metric::dh1_dq2(grid.x(i), grid.y(j)) -
                                           u2 * inv_h1h2 * Metric::dh2_dq1(grid.x(i), grid.y(j)));
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
  // T11 = FUX
  // T12 = FUY
  // T21 = FVX
  // T22 = FVY

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const auto dh2T11_dq1 = (Metric::h2(grid.xm(i), grid.ym(j)) * FUX(i, j) -
                             Metric::h2(grid.xm(i - 1), grid.ym(j)) * FUX(i - 1, j)) /
                            grid.dx();
    const auto dh1T21_dq2 = (Metric::h1(grid.x(i), grid.y(j + 1)) * FUY(i, j + 1) -
                             Metric::h1(grid.x(i), grid.y(j)) * FUY(i, j)) /
                            grid.dy();
    const auto T12        = (FVX(i, j + 1) + FVX(i, j)) / 2.0;
    const auto T22        = (FVY(i, j) + FVY(i - 1, j)) / 2.0;

    const auto inv_h1h2 =
        1.0 / (Metric::h1(grid.x(i), grid.ym(j)) * Metric::h2(grid.x(i), grid.ym(j)));

    u.x(i, j) = u_old.x(i, j) + dt * inv_h1h2 *
                                    (dh2T11_dq1 + dh1T21_dq2 +                       //
                                     T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) +  //
                                     T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j)));
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const auto dh2T12_dq1 = (Metric::h2(grid.x(i + 1), grid.y(j)) * FVX(i + 1, j) -
                             Metric::h2(grid.x(i), grid.y(j)) * FVX(i, j)) /
                            grid.dx();
    const auto dh1T22_dq2 = (Metric::h1(grid.xm(i), grid.ym(j)) * FVY(i, j) -
                             Metric::h1(grid.xm(i), grid.ym(j - 1)) * FVY(i, j - 1)) /
                            grid.dy();
    const auto T21        = (FVX(i + 1, j) + FVX(i, j)) / 2.0;
    const auto T11        = (FUX(i, j) + FUX(i, j - 1)) / 2.0;

    const auto inv_h1h2 =
        1.0 / (Metric::h1(grid.xm(i), grid.y(j)) * Metric::h2(grid.xm(i), grid.y(j)));

    u.y(i, j) = u_old.y(i, j) + dt * inv_h1h2 *
                                    (dh2T12_dq1 + dh1T22_dq2 +                       //
                                     T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -  //
                                     T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j)));
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
