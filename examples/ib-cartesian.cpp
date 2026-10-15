#include <charconv>

#include <poisfft.h>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Advection-Diffusion.hpp"
#include "AslamExtrapolation.hpp"
#include "BoundaryConditions.hpp"
#include "Common.hpp"
#include "Geometry.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IB.hpp"
#include "IO.hpp"
#include "Mac.hpp"
#include "Monitor.hpp"
#include "Quadrature.hpp"

using Float              = double;

constexpr Float x_min    = 0.0;
constexpr Float x_max    = 1.0;
constexpr Float y_min    = 0.0;
constexpr Float y_max    = 1.0;

constexpr Float rho      = 1.0;
constexpr Float mu       = 1e-3;
constexpr Float Uavg     = 1.0;
constexpr Float D        = 1e-3;

constexpr Float CFL      = 0.7;
constexpr Float tend     = 0.6;
constexpr Float dt_write = 0.001;

constexpr Circle<Float> wall{.x = 0.5, .y = 0.5, .r = 0.125};

// =================================================================================================
constexpr auto inlet_u(Float y) -> Float {
  constexpr Float H = y_max - y_min;
  const Float s     = (y - y_min) / H;
  return Uavg * 6.0 * s * (1.0 - s);
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
void correct_outflow(const Grid<Float, LAYOUT>& grid, FaceVector<Float, LAYOUT> u) {
  // Rescale the outflow so that it matches the inflow exactly (global continuity).
  Float Qin  = 0.0;
  Float Qout = 0.0;
  for (Index j = 0; j < u.x.ny(); ++j) {
    Qin  += u.x(0, j) * grid.dy();
    Qout += u.x(u.x.nx() - 1, j) * grid.dy();
  }
  const Float corr = (Qin - Qout) / (static_cast<Float>(u.x.ny()) * grid.dy());
  for (Index j = 0; j < u.x.ny(); ++j) {
    u.x(u.x.nx() - 1, j) += corr;
  }
}

#if 0
// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void correct_scalar_ib_implicit_euler(const Grid<Float, LAYOUT>& grid,
                                                const Scalar<Float, LAYOUT> ib_corr,
                                                Float D_,
                                                Float dt,
                                                Scalar<Float, LAYOUT> s) {
  grid.foreach_i(FOREACH_FUNC { s(i, j) /= 1.0 + dt * D_ * ib_corr(i, j); });
}

// =================================================================================================
namespace OrthogonalCoordinates {

template <typename Metric,
          typename Float,
          Layout LAYOUT,
          IsNoneOr<Scalar<Float, LAYOUT>> J_t,
          IsNoneOr<Scalar<Float, LAYOUT>> SRC_t>
constexpr void update_s_ib_semi_analytical(const Grid<Float, LAYOUT>& grid,
                                           Float dt,
                                           const J_t J_old,
                                           const J_t J,
                                           const FaceVector<Float, LAYOUT> F,
                                           const SRC_t src,
                                           Float D_,
                                           const Scalar<Float, LAYOUT> s_old,
                                           Scalar<Float, LAYOUT> s) {
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

  // grid.foreach_i(FOREACH_FUNC {
  //   const auto H_right  = Metric::H(grid.x(i + 1), grid.ym(j));
  //   const auto h1_right = Metric::h1(grid.x(i + 1), grid.ym(j));
  //   const auto H_left   = Metric::H(grid.x(i), grid.ym(j));
  //   const auto h1_left  = Metric::h1(grid.x(i), grid.ym(j));
  //   const auto dFdq1 =
  //       (H_right / h1_right * F.right(i, j) - H_left / h1_left * F.left(i, j)) / grid.dx();

  //   const auto H_top     = Metric::H(grid.xm(i), grid.y(j + 1));
  //   const auto h2_top    = Metric::h2(grid.xm(i), grid.y(j + 1));
  //   const auto H_bottom  = Metric::H(grid.xm(i), grid.y(j));
  //   const auto h2_bottom = Metric::h2(grid.xm(i), grid.y(j));
  //   const auto dFdq2 =
  //       (H_top / h2_top * F.top(i, j) - H_bottom / h2_bottom * F.bottom(i, j)) / grid.dy();

  //   const auto H          = Metric::H(grid.xm(i), grid.ym(j));
  //   const auto Ji_old     = IF_NONE_ELSE(J_t, 1.0, J_old(i, j));
  //   const auto Ji         = IF_NONE_ELSE(J_t, 1.0, J(i, j));
  //   const auto div_factor = IF_NONE_ELSE(J_t, 1.0 / H, 1.0);
  //   const auto src_factor = IF_NONE_ELSE(J_t, 1.0, H);

  //   const auto srci       = IF_NONE_ELSE(SRC_t, 0.0, src(i, j));

  //   s(i, j) = (Ji_old * s_old(i, j) - dt * (div_factor * (dFdq1 + dFdq2) - src_factor * srci)) /
  //   Ji;
  // });

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

template <typename Float,
          Layout LAYOUT,
          IsNoneOr<Scalar<Float, LAYOUT>> J_t,
          IsNoneOr<Scalar<Float, LAYOUT>> FWZ_t>
constexpr void update_s_ib_semi_analytical(const Grid<Float, LAYOUT>& grid,
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
          OrthogonalCoordinates::update_s_ib_semi_analytical<Metric>(
              grid, dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, mu, rho, ib_corr, u_old, u);
        }
      });
}
#endif

// =================================================================================================
auto main(int argc, char** argv) -> int {
  const auto usage_str = Igor::detail::format("Usage: {} <grid size>", argv[0]);
  if (argc < 2) {
    Igor::Error("{}", usage_str);
    return 1;
  }

  Index N = 0;
  if (std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), N).ec != std::errc{} || N <= 0) {
    Igor::Error("{}", usage_str);
    Igor::Error("  Invalid grid size `{}`", argv[1]);
    return 1;
  }

  const auto output_dir = get_output_directory();
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(x_min, x_max, N, y_min, y_max, N, 3);

  auto u_old = grid.alloc_face_vector();
  auto u     = grid.alloc_face_vector();
  auto ui    = grid.alloc_vector();

  auto FUX   = grid.alloc_scalar();
  auto FUY   = grid.alloc_vertex_scalar();
  auto FVX   = grid.alloc_vertex_scalar();
  auto FVY   = grid.alloc_scalar();

  auto p     = grid.alloc_scalar();
  auto dp    = grid.alloc_scalar();
  auto div   = grid.alloc_scalar();

  auto s     = grid.alloc_scalar();
  auto s_old = grid.alloc_scalar();
  auto FS    = grid.alloc_face_vector();

  // auto ib_corr_center = grid.alloc_scalar();
  auto ib_corr_face = grid.alloc_face_vector();
  auto ib_levelset  = grid.alloc_scalar();
  auto ib_free      = grid.alloc_scalar();

  Float dt          = 0.0;
  Float t           = 0.0;

  grid.foreach_i(FOREACH_FUNC {
    const auto x      = grid.xm(i);
    const auto y      = grid.ym(j);
    ib_levelset(i, j) = wall.levelset(x, y);
    ib_free(i, j) =
        1.0 - quadrature([](Float x, Float y) { return wall.contains({.x = x, .y = y}); },
                         grid.x(i),
                         grid.x(i + 1),
                         grid.y(j),
                         grid.y(j + 1)) /
                  grid.dv(i, j);
  });

  // = Linear solver ===============================================================================
  const std::array<int, 2> ns   = {grid.nx(), grid.ny()};
  const std::array<Float, 2> Ls = {grid.x_max() - grid.x_min(), grid.y_max() - grid.y_min()};
  const std::array<int, 4> BCs  = {
      PoisFFT::NEUMANN_STAG,
      PoisFFT::NEUMANN_STAG,
      PoisFFT::NEUMANN_STAG,
      PoisFFT::NEUMANN_STAG,
  };
  PoisFFT::Solver<2, Float> solver(ns.data(), Ls.data(), BCs.data(), PoisFFT::FINITE_DIFFERENCE_2);
  const std::array<int, 2> ngs = {grid.nghost(), grid.nghost()};
  // = Linear solver ===============================================================================

  const BConds<Float> u_bconds{
      .left = Dirichlet<Float>{.val = [](Float y, Float /*t*/) { return inlet_u(y); }},
      // .left   = Dirichlet<Float>{.val = 0.0},
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Dirichlet<Float>{.val = 0.0},
  };

  const BConds<Float> v_bconds{
      .left   = Dirichlet<Float>{.val = 0.0},
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Dirichlet<Float>{.val = 0.0},
  };
  // const BConds<Float> s_bconds{
  //     .left =
  //         Dirichlet<Float>{
  //             .val = [](Float y,
  //                       Float /*t*/) { return static_cast<Float>(0.25 <= y && y <= 0.75); },
  //         },
  //     .right  = Neumann(),
  //     .bottom = Neumann(),
  //     .top    = Neumann(),
  // };
  const BConds<Float> s_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };
  const auto ib_dsdn = 0.0;

  grid.foreach_i(FOREACH_FUNC {
    constexpr Circle<Float> c{.x = 0.125, .y = 0.5, .r = 0.0625};
    const auto x = grid.xm(i);
    const auto y = grid.ym(j);
    s(i, j)      = c.contains(x, y);
    // s(i, j) = 0.0;
  });
  apply_bconds(grid, s_bconds, s, t);

  grid.foreach_face_i<Dimension::X>(FOREACH_FUNC { u.x(i, j) = 0.0; });
  grid.foreach_face_i<Dimension::Y>(FOREACH_FUNC { u.y(i, j) = 0.0; });
  apply_velocity_bconds(grid, u_bconds, v_bconds, u);
  interpolate(grid, u, ui);

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("s", s);
  writer.add_field("div", div);
  writer.add_field("psi", ib_levelset);
  if (!writer.write(t)) { return 1; }

  Stats p_stats   = stats(grid, p);
  Stats u_stats   = stats(grid, u.x);
  Stats v_stats   = stats(grid, u.y);
  Stats s_stats   = stats(grid, s, ib_free);
  Stats div_stats = stats(grid, div);
  Float div_max   = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.max, "max(u)");
  monitor.add_variable(&v_stats.max, "max(v)");
  monitor.add_variable(&s_stats.min, "min(s)");
  monitor.add_variable(&s_stats.max, "max(s)");
  monitor.add_variable(&s_stats.sum, "sum(s)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.write();

  calc_ib_correction_shape(grid, wall, ib_corr_face);

  IGOR_TIME_SCOPE("Solver")
  while (t < tend) {
    dt = std::min({
        adjust_dt(grid, u, rho, mu, CFL),
        advection_adjust_dt(grid, D, CFL),
        dt_write,
        tend - t,
    });

    copy(u, u_old);
    copy(s, s_old);

    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      // 1) Predictor
      calc_mom_flux(grid, u, p, rho, mu, FUX, FUY, FVX, FVY);
#if 1
      update_u(grid, local_dt, FUX, FUY, FVX, FVY, u_old, u);
      correct_velocity_ib_implicit_euler(grid, ib_corr_face, rho, mu, local_dt, u);
#else
      update_u_ib_semi_analytical(grid,
                                  local_dt,
                                  None{},
                                  None{},
                                  FUX,
                                  FUY,
                                  FVX,
                                  FVY,
                                  None{},
                                  mu,
                                  rho,
                                  ib_corr_face,
                                  u_old,
                                  u);
#endif
      apply_velocity_bconds(grid, u_bconds, v_bconds, u);
      correct_outflow(grid, u);

      // 2) Pressure correction
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rho / local_dt; });
      solver.execute(dp.data(), div.data(), ngs.data(), ngs.data());
      apply_neumann_bconds(grid, dp);

      // 3) Project
      correct_velocity(grid, dp, rho, local_dt, u, p);

      // 4) Update scalar
#if 0
      calc_advection_flux(grid, u, s, D, FS);
#if 1
      update_s(grid, local_dt, FS, s_old, s);
      correct_scalar_ib_implicit_euler(grid, ib_corr_center, D, local_dt, s);
#else
      update_s_ib_semi_analytical(grid,
                                  local_dt,
                                  None{},
                                  None{},
                                  FUX,
                                  FUY,
                                  FVX,
                                  FVY,
                                  None{},
                                  mu,
                                  rho,
                                  ib_corr_center,
                                  u_old,
                                  u);
#endif
      apply_bconds(grid, s_bconds, s, t);
#else
      const auto band_size = 4.0 * 6.0 * std::max(grid.dx(), grid.dy());
      neumann_extrapolate<2>(grid, s, ib_levelset, ib_dsdn, band_size);
      calc_advection_flux(grid, u, s, D, FS);
      update_s(grid, local_dt, FS, s_old, s);
      apply_bconds(grid, s_bconds, s, t);
#endif
    }

    interpolate(grid, u, ui);
    calc_div(grid, u, div);

    p_stats    = stats(grid, p);
    u_stats    = stats(grid, u.x);
    v_stats    = stats(grid, u.y);
    s_stats    = stats(grid, s, ib_free);
    div_stats  = stats(grid, div);
    div_max    = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

    t         += dt;
    if (should_save(t, dt, dt_write, tend)) {
      if (!writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Ok.");
}
