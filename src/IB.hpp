#pragma once

#include <cstdint>

#include <Igor/Logging.hpp>

#include "Grid.hpp"
#include "Metrics.hpp"

// See: Luchini, P., Gatti, D., Chiarini, A., Gattere, F., Atzori, M., Quadrio, M., 2025. A simple
//      and efficient second-order immersed-boundary method for the incompressible Navier–Stokes
//      equations. Journal of Computational Physics 539, 114245.
//      https://doi.org/10.1016/j.jcp.2025.114245

// -------------------------------------------------------------------------------------------------
enum Flow : uint8_t {
  FREE_FLOW      = 0b00000,
  WALL_TO_LEFT   = 0b00001,
  WALL_TO_RIGHT  = 0b00010,
  WALL_TO_BOTTOM = 0b00100,
  WALL_TO_TOP    = 0b01000,
  SOLID          = 0b10000,
};

template <typename FUNC, typename Float, Layout LAYOUT>
constexpr auto
characterize_flow_regime(const Grid<Float, LAYOUT>& grid, FUNC immersed_wall, Float x, Float y)
    -> uint8_t {
  if (immersed_wall(x, y) > 0.0) { return Flow::SOLID; }

  uint8_t flow = Flow::FREE_FLOW;
  if (immersed_wall(x + grid.dx(), y) > 0.0) { flow |= Flow::WALL_TO_RIGHT; }
  if (immersed_wall(x - grid.dx(), y) > 0.0) { flow |= Flow::WALL_TO_LEFT; }
  if (immersed_wall(x, y + grid.dy()) > 0.0) { flow |= Flow::WALL_TO_TOP; }
  if (immersed_wall(x, y - grid.dy()) > 0.0) { flow |= Flow::WALL_TO_BOTTOM; }
  return flow;
}

// -------------------------------------------------------------------------------------------------
template <typename Shape, typename Float, Layout LAYOUT>
constexpr void calc_ib_correction_shape(const Grid<Float, LAYOUT>& grid,
                                        const Shape& wall,
                                        FaceVector<Float, LAYOUT> ib_corr) {
  using TransformFunc = Vec2<Float> (*)(const Vec2<Float>&);
  TransformFunc to_cartesian;
  switch (grid.coords()) {
    case Coordinates::CARTESIAN: to_cartesian = [](const Vec2<Float>& xy) { return xy; }; break;
    case Coordinates::POLAR:
    case Coordinates::SYMMETRIC_SPHERICAL: to_cartesian = &polar2cartesian; break;
  }

  auto immersed_wall = [&](Float x, Float y) {
    return wall.contains(to_cartesian(Vec2<Float>{.x = x, .y = y}));
  };

  const auto dx  = grid.dx();
  const auto dy  = grid.dy();

  auto calc_corr = [=](const Vec2<Float>& p_center, Float& corr) {
    const auto flow = characterize_flow_regime(grid, immersed_wall, p_center.x, p_center.y);
    if (flow == Flow::FREE_FLOW) { return; }
    if (flow == Flow::SOLID) {
      corr = std::numeric_limits<Float>::max();
      return;
    }

    [[maybe_unused]] constexpr Float TOL = 1e-6;
    if ((flow & Flow::WALL_TO_RIGHT) > 0) {
      const Vec2<Float> p_other   = {.x = p_center.x + dx, .y = p_center.y};
      const Vec2<Float> intersect = wall.intersect_line(p_center, p_other);
      const Float dist            = intersect.x - p_center.x;
      IGOR_ASSERT(0.0 < dist && dist < dx + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dx,
                  dist);
      const Float lambda  = (dx - dist) / (dist * dx * dx);
      corr               += lambda;
    }
    if ((flow & Flow::WALL_TO_LEFT) > 0) {
      const Vec2<Float> p_other   = {.x = p_center.x - dx, .y = p_center.y};
      const Vec2<Float> intersect = wall.intersect_line(p_center, p_other);
      const Float dist            = p_center.x - intersect.x;
      IGOR_ASSERT(0.0 < dist && dist < dx + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dx,
                  dist);
      const Float lambda  = (dx - dist) / (dist * dx * dx);
      corr               += lambda;
    }
    if ((flow & Flow::WALL_TO_TOP) > 0) {
      const Vec2<Float> p_other   = {.x = p_center.x, .y = p_center.y + dy};
      const Vec2<Float> intersect = wall.intersect_line(p_center, p_other);
      const Float dist            = intersect.y - p_center.y;
      IGOR_ASSERT(0.0 < dist && dist < dy + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dy,
                  dist);
      const Float lambda  = (dy - dist) / (dist * dy * dy);
      corr               += lambda;
    }
    if ((flow & Flow::WALL_TO_BOTTOM) > 0) {
      const Vec2<Float> p_other   = {.x = p_center.x, .y = p_center.y - dy};
      const Vec2<Float> intersect = wall.intersect_line(p_center, p_other);
      const Float dist            = p_center.y - intersect.y;
      IGOR_ASSERT(0.0 < dist && dist < dy + TOL,
                  "Expected dist in [0, {:.6e}] but got dist = {:.6e}",
                  dy,
                  dist);
      const Float lambda  = (dy - dist) / (dist * dy * dy);
      corr               += lambda;
    }
  };

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const Vec2<Float> p_center = {
        .x = grid.x(i),
        .y = grid.ym(j),
    };
    calc_corr(p_center, ib_corr.x(i, j));
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const Vec2<Float> p_center = {
        .x = grid.xm(i),
        .y = grid.y(j),
    };
    calc_corr(p_center, ib_corr.y(i, j));
  });
}

// -------------------------------------------------------------------------------------------------
template <typename Float, Layout LAYOUT>
constexpr void correct_velocity_ib_implicit_euler(const Grid<Float, LAYOUT>& grid,
                                                  const FaceVector<Float, LAYOUT> ib_corr,
                                                  Float rho,
                                                  Float mu,
                                                  Float dt,
                                                  FaceVector<Float, LAYOUT> u) {
  const auto nu = mu / rho;
  grid.template foreach_face_i<Dimension::X>(
      FOREACH_FUNC { u.x(i, j) /= 1.0 + dt * nu * ib_corr.x(i, j); });
  grid.template foreach_face_i<Dimension::Y>(
      FOREACH_FUNC { u.y(i, j) /= 1.0 + dt * nu * ib_corr.y(i, j); });
}

// -------------------------------------------------------------------------------------------------
template <typename Float,
          Layout LAYOUT,
          IsNoneOr<Scalar<Float, LAYOUT>> J_t,
          IsNoneOr<Scalar<Float, LAYOUT>> FWZ_t>
constexpr void update_u_ib_semi_analytical(const Grid<Float, LAYOUT>& grid,
                                           Float dt,
                                           const J_t J_old,
                                           const J_t J,
                                           const Scalar<Float, LAYOUT> FUX,
                                           const VertexScalar<Float, LAYOUT> FUY,
                                           const VertexScalar<Float, LAYOUT> FVX,
                                           const Scalar<Float, LAYOUT> FVY,
                                           const FWZ_t FWZ,
                                           Float mu,
                                           Float rho,
                                           const FaceVector<Float, LAYOUT> ib_corr,
                                           const FaceVector<Float, LAYOUT> u_old,
                                           FaceVector<Float, LAYOUT> u) {
  using Metric = Metric::Cartesian;
  static_assert((Metric::is_2d && IsNone<FWZ_t>) || (!Metric::is_2d && !IsNone<FWZ_t>),
                "Provide `Scalar` for `FWZ` in quasi-3D case and `None` in 2D case.");
  static_assert(IsNone<J_t>, "ALE is not yet implemented.");
  static_cast<void>(J_old);
  static_cast<void>(J);

  // From Luchini et al.
  // A*U^(n+1) - B*U^n = C*F^n
  // U^(n+1) = (C*F^n + B*U^n) / A
  //
  // B = (lambda*dt) / (exp(lambda*dt) - 1)
  // A = lambda*dt + B
  // C = dt

  auto calc_coefficients = [](Float lambda, Float dt, Float& A, Float& B, Float& C) {
    B = std::abs(lambda) < 1e-6 ? 1.0 : (lambda * dt) / (std::exp(lambda * dt) - 1.0);
    IGOR_ASSERT(
        0 <= B && B <= 1, "Expected B in [0, 1] but got B = {:.6e} (lambda = {:.6e})", B, lambda);
    A = lambda * dt + B;
    C = dt;
  };
  constexpr Float LAMBDA_MAX = 1e10;

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
    const auto T33   = IF_NONE_ELSE(FWZ_t, 0.0, (FWZ(i, j) + FWZ(i - 1, j)) / 2.0);

    const auto h2    = Metric::h2(grid.x(i), grid.ym(j));
    const auto h3    = Metric::h3(grid.x(i), grid.ym(j));
    const auto inv_H = 1.0 / Metric::H(grid.x(i), grid.ym(j));

    // const auto Ji_old     = IF_NONE_ELSE(J_t, 1.0, (J_old(i, j) + J_old(i - 1, j)) / 2.0);
    // const auto Ji         = IF_NONE_ELSE(J_t, 1.0, (J(i, j) + J(i - 1, j)) / 2.0);
    const auto div_factor = IF_NONE_ELSE(J_t, inv_H, 1.0);

    const auto dudt       = -div_factor * (dHT11_dq1 + dHT21_dq2 +  //
                                           h3 * T12 * Metric::dh1_dq2(grid.x(i), grid.ym(j)) -
                                           h3 * T22 * Metric::dh2_dq1(grid.x(i), grid.ym(j)) -
                                           h2 * T33 * Metric::dh3_dq1(grid.x(i), grid.ym(j)));

    const Float lambda    = mu / rho * ib_corr.x(i, j);
    if (lambda > LAMBDA_MAX) {
      u.x(i, j) = 0.0;
      return;
    }

    Float A, B, C;  // NOLINT
    calc_coefficients(lambda, dt, A, B, C);
    u.x(i, j) = (B * u_old.x(i, j) + C * dudt) / A;
    // u.x(i, j)       = (Ji_old * u_old.x(i, j) + dt * dudt) / Ji;
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
    const auto T21   = (FUY(i + 1, j) + FUY(i, j)) / 2.0;
    const auto T33   = IF_NONE_ELSE(FWZ_t, 0.0, (FWZ(i, j) + FWZ(i, j - 1)) / 2.0);

    const auto h1    = Metric::h1(grid.xm(i), grid.y(j));
    const auto h3    = Metric::h3(grid.xm(i), grid.y(j));
    const auto inv_H = 1.0 / Metric::H(grid.xm(i), grid.y(j));

    // const auto Ji_old     = IF_NONE_ELSE(J_t, 1.0, (J_old(i, j) + J_old(i, j - 1)) / 2.0);
    // const auto Ji         = IF_NONE_ELSE(J_t, 1.0, (J(i, j) + J(i, j - 1)) / 2.0);
    const auto div_factor = IF_NONE_ELSE(J_t, inv_H, 1.0);

    const auto dvdt       = -div_factor * (dHT12_dq1 + dHT22_dq2 +  //
                                           h3 * T21 * Metric::dh2_dq1(grid.xm(i), grid.y(j)) -
                                           h3 * T11 * Metric::dh1_dq2(grid.xm(i), grid.y(j)) -
                                           h1 * T33 * Metric::dh3_dq2(grid.xm(i), grid.y(j)));

    const Float lambda    = mu / rho * ib_corr.y(i, j);
    if (lambda > LAMBDA_MAX) {
      u.y(i, j) = 0.0;
      return;
    }

    Float A, B, C;  // NOLINT
    calc_coefficients(lambda, dt, A, B, C);
    u.y(i, j) = (B * u_old.y(i, j) + C * dvdt) / A;
    // u.y(i, j)       = (Ji_old * u_old.y(i, j) + dt * dvdt) / Ji;
  });
}
