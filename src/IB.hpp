#pragma once

#include <cstdint>
#include <limits>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "Grid.hpp"
#include "MacOrthogonal.hpp"
#include "Metrics.hpp"
#include "Quadrature.hpp"

// See: Luchini, P., Gatti, D., Chiarini, A., Gattere, F., Atzori, M., Quadrio, M., 2025. A simple
//      and efficient second-order immersed-boundary method for the incompressible Navier–Stokes
//      equations. Journal of Computational Physics 539, 114245.
//      https://doi.org/10.1016/j.jcp.2025.114245
//
// The viscous term at a velocity node P next to the wall contains a_N * u_N for a neighbour N
// inside the solid. Luchini et al. replace u_N by the value linearly extrapolated through the wall
// (u = 0):
//
//   u_N* = -u_P * (1 - delta) / delta,
//
// where delta in (0, 1] is the fraction of the step P -> N at which the wall is crossed. With u_N =
// 0 inside the solid this is a diagonal term -nu * lambda * u_P with
//
//   lambda = sum_N a_N * (1 - delta) / delta.
//
// In orthogonal coordinates a_N is the weight of N in the discrete scalar Laplacian
//
//   (1/H) d/dq1 (H/h1^2 df/dq1) + (1/H) d/dq2 (H/h2^2 df/dq2).
//
// The momentum flux is in stress form, which reduces to the vector Laplacian for a divergence-free
// velocity, so these are the weights of the Laplacian and not the ones of `FUX` or `FVY`. The
// curvature terms of the vector Laplacian are diagonal or first-derivative terms and are not
// corrected.
//
// Instead of linearly in q, the ghost value is extrapolated linearly in the harmonic coordinate xi
// with dxi/dq = h^2 / H, for which the 1D operator (1/H) d/dq (H/h^2 df/dq) has the exact solutions
// f = a + b * xi, e.g. xi = ln(r) for r in polar and xi = -1/r for r in spherical coordinates:
//
//   u_N* = -u_P * (xi_w - xi_N) / (xi_P - xi_w).
//
// This is (1 - delta) / delta in Cartesian coordinates and for theta in polar coordinates. As in
// the Cartesian case, the remaining local error is then proportional to the operator at the wall,
// which vanishes e.g. for Couette flow. Plain linear extrapolation in r leaves an additional error
// proportional to the wall shear divided by r.

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
// Fraction delta in (0, 1] of the step p -> n at which the wall is crossed
template <typename FUNC, typename Float>
[[nodiscard]] constexpr auto
wall_fraction(FUNC immersed_wall, const Vec2<Float>& p, const Vec2<Float>& n) -> Float {
  IGOR_ASSERT(!(immersed_wall(p.x, p.y) > 0.0) && immersed_wall(n.x, n.y) > 0.0,
              "Expected p = ({:.6e}, {:.6e}) outside and n = ({:.6e}, {:.6e}) inside the wall.",
              p.x,
              p.y,
              n.x,
              n.y);

  constexpr int MAX_ITER = 64;
  Float lo               = 0.0;
  Float hi               = 1.0;
  // Bisection
  for (int iter = 0; iter < MAX_ITER; ++iter) {
    const Float mid = (lo + hi) / 2.0;
    if (immersed_wall(p.x + mid * (n.x - p.x), p.y + mid * (n.y - p.y)) > 0.0) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return hi;
}

// -------------------------------------------------------------------------------------------------
// Weight of the neighbour at q1 + sign * dq1 in the discrete scalar Laplacian at p.
template <typename Metric, typename Float>
[[nodiscard]] constexpr auto laplace_weight_q1(const Vec2<Float>& p, Float sign, Float dq1)
    -> Float {
  const Float q1_face = p.x + sign * dq1 / 2.0;
  return Metric::H(q1_face, p.y) / Igor::sqr(Metric::h1(q1_face, p.y)) /
         (Metric::H(p.x, p.y) * Igor::sqr(dq1));
}

// Weight of the neighbour at q2 + sign * dq2 in the discrete scalar Laplacian at p.
template <typename Metric, typename Float>
[[nodiscard]] constexpr auto laplace_weight_q2(const Vec2<Float>& p, Float sign, Float dq2)
    -> Float {
  const Float q2_face = p.y + sign * dq2 / 2.0;
  return Metric::H(p.x, q2_face) / Igor::sqr(Metric::h2(p.x, q2_face)) /
         (Metric::H(p.x, p.y) * Igor::sqr(dq2));
}

// Ghost value factor (xi_w - xi_N) / (xi_P - xi_w), i.e. u_N* = -u_P * factor, for the wall at the
// fraction delta of the step from p to n, which differ in exactly one coordinate.
template <typename Metric, typename Float>
[[nodiscard]] constexpr auto ghost_factor(const Vec2<Float>& p, const Vec2<Float>& n, Float delta)
    -> Float {
  // dxi/dt along p + t * (n - p), up to the constant factor dq/dt that cancels in the ratio.
  auto dxi_dt = [&](Float t) -> Float {
    const Float q1 = p.x + t * (n.x - p.x);
    const Float q2 = p.y + t * (n.y - p.y);
    const Float h  = p.x != n.x ? Metric::h1(q1, q2) : Metric::h2(q1, q2);
    return Igor::sqr(h) / Metric::H(q1, q2);
  };
  return quadrature<4>(dxi_dt, delta, Float{1}) / quadrature<4>(dxi_dt, Float{0}, delta);
}

// -------------------------------------------------------------------------------------------------
template <typename Shape, typename Float, Layout LAYOUT>
constexpr void calc_ib_correction_shape(const Grid<Float, LAYOUT>& grid,
                                        const Shape& wall,
                                        FaceVector<Float, LAYOUT> ib_corr) {
  auto immersed_wall = [&](Float x, Float y) {
    switch (grid.coords()) {
      case Coordinates::CARTESIAN:           return wall.contains(Vec2<Float>{.x = x, .y = y});
      case Coordinates::POLAR:
      case Coordinates::SYMMETRIC_SPHERICAL: return wall.contains(polar2cartesian(x, y));
    }
  };

  const auto dx = grid.dx();
  const auto dy = grid.dy();

  dispatch_metric(
      grid, DISPATCH_FUNC {
        auto calc_corr = [=](const Vec2<Float>& p, Float& corr) {
          const auto flow = characterize_flow_regime(grid, immersed_wall, p.x, p.y);
          if (flow == Flow::FREE_FLOW) { return; }
          if (flow == Flow::SOLID) {
            corr = std::numeric_limits<Float>::max();
            return;
          }

          // Faces on the symmetry axis (H = 0) are set by the boundary conditions.
          if (!(Metric::H(p.x, p.y) > 0.0)) { return; }

          auto add_corr = [&](Flow wall_dir, const Vec2<Float>& n, Float weight) {
            if ((flow & wall_dir) == 0) { return; }
            const Float delta = wall_fraction(immersed_wall, p, n);
            IGOR_ASSERT(0.0 < delta && delta <= 1.0,
                        "Expected delta in (0, 1] but got delta = {:.6e}",
                        delta);
            // corr += weight * (1.0 - delta) / delta;
            corr += weight * ghost_factor<Metric>(p, n, delta);
          };

          add_corr(Flow::WALL_TO_RIGHT,
                   {.x = p.x + dx, .y = p.y},
                   laplace_weight_q1<Metric>(p, Float{1}, dx));
          add_corr(Flow::WALL_TO_LEFT,
                   {.x = p.x - dx, .y = p.y},
                   laplace_weight_q1<Metric>(p, Float{-1}, dx));
          add_corr(Flow::WALL_TO_TOP,
                   {.x = p.x, .y = p.y + dy},
                   laplace_weight_q2<Metric>(p, Float{1}, dy));
          add_corr(Flow::WALL_TO_BOTTOM,
                   {.x = p.x, .y = p.y - dy},
                   laplace_weight_q2<Metric>(p, Float{-1}, dy));
        };

        grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
          const Vec2<Float> p = {
              .x = grid.x(i),
              .y = grid.ym(j),
          };
          calc_corr(p, ib_corr.x(i, j));
        });

        grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
          const Vec2<Float> p = {
              .x = grid.xm(i),
              .y = grid.y(j),
          };
          calc_corr(p, ib_corr.y(i, j));
        });
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
namespace OrthogonalCoordinates {

template <typename Metric,
          typename Float,
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
    // `expm1` avoids the cancellation in exp(x) - 1 that gives B > 1 for small lambda * dt.
    const Float x = lambda * dt;
    B             = std::abs(x) < 1e-12 ? 1.0 : x / std::expm1(x);
    IGOR_ASSERT(
        0 <= B && B <= 1, "Expected B in [0, 1] but got B = {:.6e} (lambda = {:.6e})", B, lambda);
    A = lambda * dt + B;
    C = dt;
  };
  constexpr Float LAMBDA_MAX = 1e10;

  const auto nu              = mu / rho;

  grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
    const Float lambda = nu * ib_corr.x(i, j);
    if (lambda > LAMBDA_MAX) {
      u.x(i, j) = 0.0;
      return;
    }

    const auto inv_H = 1.0 / Metric::H(grid.x(i), grid.ym(j));
    const auto dudt  = -inv_H * calc_H_div_flux_x<Metric>(grid, i, j, FUX, FUY, FVX, FVY, FWZ);

    Float A, B, C;  // NOLINT
    calc_coefficients(lambda, dt, A, B, C);
    u.x(i, j) = (B * u_old.x(i, j) + C * dudt) / A;
  });

  grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    const Float lambda = nu * ib_corr.y(i, j);
    if (lambda > LAMBDA_MAX) {
      u.y(i, j) = 0.0;
      return;
    }

    const auto inv_H = 1.0 / Metric::H(grid.xm(i), grid.y(j));
    const auto dvdt  = -inv_H * calc_H_div_flux_y<Metric>(grid, i, j, FUX, FUY, FVX, FVY, FWZ);

    Float A, B, C;  // NOLINT
    calc_coefficients(lambda, dt, A, B, C);
    u.y(i, j) = (B * u_old.y(i, j) + C * dvdt) / A;
  });
}

}  // namespace OrthogonalCoordinates

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
  dispatch_metric(
      grid, DISPATCH_FUNC {
        if constexpr (Metric::is_2d != IsNone<FWZ_t>) {
          Igor::Panic("`FWZ` must be given iff coordinates are quasi-3D");
        } else {
          OrthogonalCoordinates::update_u_ib_semi_analytical<Metric>(
              grid, dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, mu, rho, ib_corr, u_old, u);
        }
      });
}
