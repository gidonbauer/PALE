#include <charconv>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "BoundaryConditions.hpp"
#include "Common.hpp"
#include "Geometry.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IB.hpp"
#include "IO.hpp"
#include "Mac.hpp"
#include "Monitor.hpp"
#include "MultigridPoisson.hpp"

#include "Test-Common.hpp"

using Float           = double;

constexpr Float x_min = 0.0;
constexpr Float x_max = 1.0;
constexpr Float y_min = 0.0;
constexpr Float y_max = 2.0;

constexpr Float rho   = 1.0;
constexpr Float mu    = 1.0;
constexpr Float Uavg  = 1.0;
constexpr Vec2<Float> u_wall{.x = 0.0, .y = 1.0};

constexpr Float CFL      = 0.5;
constexpr Float tend     = 1.0;
constexpr Float dt_write = tend / 100.0;

HalfPlane<Float> bottom_wall{.x = 0.0, .y = 0.0, .nx = 0.0, .ny = -1.0};  // NOLINT
HalfPlane<Float> top_wall{.x = 0.0, .y = 1.0, .nx = 0.0, .ny = 1.0};      // NOLINT

// =================================================================================================
// namespace Expected {
//
// constexpr std::array ns  = {16, 32, 64, 128, 256};
// constexpr std::array L1s = {
//     2.11220054e-03,
//     5.32794280e-04,
//     1.33995883e-04,
//     3.35113462e-05,
//     8.37842234e-06,
// };
// static_assert(ns.size() == L1s.size());
// constexpr auto L1(Index n) { return interp_n2(ns, L1s, n); }
//
// }  // namespace Expected

// =================================================================================================
template <typename Float>
constexpr auto u_analytical(Float y) -> Float {
  if (bottom_wall.y > y || y > top_wall.y) { return 0.0; }
  const Float H = top_wall.y - bottom_wall.y;
  const Float s = (y - bottom_wall.y) / H;
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

  const std::string output_dir = "./test/output/IB-Moving-Channel-" + std::to_string(N) + '/';
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(x_min, x_max, N, y_min, y_max, N, 1);

  auto u_old       = grid.alloc_face_vector();
  auto u           = grid.alloc_face_vector();
  auto ui          = grid.alloc_vector();

  auto FUX         = grid.alloc_scalar();
  auto FUY         = grid.alloc_vertex_scalar();
  auto FVX         = grid.alloc_vertex_scalar();
  auto FVY         = grid.alloc_scalar();

  auto p           = grid.alloc_scalar();
  auto dp          = grid.alloc_scalar();
  auto div         = grid.alloc_scalar();

  auto ib_levelset = grid.alloc_scalar();
  auto ib_corr     = grid.alloc_face_vector();

  auto u_exp       = grid.alloc_vector();
  grid.foreach_a(FOREACH_FUNC {
    u_exp.x(i, j) = u_analytical(grid.ym(j));
    u_exp.y(i, j) = u_wall.y;
  });

  Float dt = 0.0;
  Float t  = 0.0;

  MultigridSolver solver(grid);
  Float mg_res    = 0.0;
  Index mg_cycles = 0;

  calc_ib_correction_shape(grid, bottom_wall, ib_corr);
  calc_ib_correction_shape(grid, top_wall, ib_corr);
  grid.foreach_a(FOREACH_FUNC {
    ib_levelset(i, j) = std::max(bottom_wall.levelset(grid.xm(i), grid.ym(j)),
                                 top_wall.levelset(grid.xm(i), grid.ym(j)));
  });

  const BConds<Float> u_bconds{
      .left   = Dirichlet<Float>{.val = [](Float y, Float /*t*/) { return u_analytical(y); }},
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Dirichlet<Float>{.val = 0.0},
  };
  const BConds<Float> v_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Dirichlet<Float>{.val = 0.0},
  };
  grid.foreach_face_i<Dimension::X>(FOREACH_FUNC { u.x(i, j) = u_analytical(grid.ym(j)); });
  interpolate(grid, u, ui);

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("div", div);
  writer.add_field("ua", u_exp);
  writer.add_field("psi", ib_levelset);
  if (!writer.write(t)) { return 1; }

  Stats p_stats   = stats(grid, p);
  Stats u_stats   = stats(grid, u.x);
  Stats v_stats   = stats(grid, u.y);
  Stats div_stats = stats(grid, div);
  Float div_max   = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.min, "min(u)");
  monitor.add_variable(&u_stats.max, "max(u)");
  monitor.add_variable(&v_stats.min, "min(v)");
  monitor.add_variable(&v_stats.max, "max(v)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.add_variable(&mg_res, "residual(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();

  IGOR_TIME_SCOPE("Solver")
  while (t < tend) {
    dt = std::min({
        adjust_dt(grid, u, rho, mu, CFL),
        dt_write,
        tend - t,
    });

    copy(u, u_old);

    mg_cycles = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      // 1) Predictor
      calc_mom_flux(grid, u, p, rho, mu, FUX, FUY, FVX, FVY);
      update_u(grid, local_dt, FUX, FUY, FVX, FVY, u_old, u);
      correct_velocity_ib_implicit_euler(grid, ib_corr, rho, mu, local_dt, u_wall, u);
      apply_velocity_bconds(grid, u_bconds, v_bconds, u);
      correct_outflow(grid, u);

      // 2) Pressure correction
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rho / local_dt; });
      solver.solve(dp, div, 1e-6 / local_dt);
      mg_res     = solver.res();
      mg_cycles += solver.num_cycles();
      apply_neumann_bconds(grid, dp);

      // 3) Project
      correct_velocity(grid, dp, rho, local_dt, u, p);

      // 4) Move immersed boundaries
      bottom_wall.y += 0.5 * dt * u_wall.y;
      top_wall.y    += 0.5 * dt * u_wall.y;
      fill(ib_corr, 0.0);
      calc_ib_correction_shape(grid, bottom_wall, ib_corr);
      calc_ib_correction_shape(grid, top_wall, ib_corr);
      grid.foreach_a(FOREACH_FUNC {
        ib_levelset(i, j) = std::max(bottom_wall.levelset(grid.xm(i), grid.ym(j)),
                                     top_wall.levelset(grid.xm(i), grid.ym(j)));
      });
    }

    interpolate(grid, u, ui);
    calc_div(grid, u, div);
    grid.foreach_a(FOREACH_FUNC {
      u_exp.x(i, j) = u_analytical(grid.ym(j));
      u_exp.y(i, j) = u_wall.y;
    });

    p_stats    = stats(grid, p);
    u_stats    = stats(grid, u.x);
    v_stats    = stats(grid, u.y);
    div_stats  = stats(grid, div);
    div_max    = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

    t         += dt;
    if (should_save(t, dt, dt_write, tend)) {
      if (!writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  // const auto L1 = grid.transform_reduce_range(
  //     u.x.nx() / 2,
  //     u.x.nx() / 2 + 1,
  //     0,
  //     u.x.ny(),
  //     Vec2<Float>{},
  //     FOREACH_FUNC {
  //       if (wall.contains(grid.x(i), grid.ym(j))) { return Vec2<Float>{}; }
  //       return Vec2<Float>{
  //                  .x = std::abs(u_exp.x(i, j) - u.x(i, j)),
  //                  .y = std::abs(u.y(i, j)),
  //              } *
  //              grid.dy();
  //     },
  //     std::plus<>{});
  // Igor::Info("L1_u({}) = {:.12e}", N, L1.x);
  // Igor::Info("L1_v({}) = {:.12e}", N, L1.y);
  // if (L1 > 1.1 * Expected::L1(N) || std::isnan(L1)) {
  //   Igor::Error("u error does not match expected value: expected {:.8e} but got {:.8e}",
  //               Expected::L1(N),
  //               L1);
  //   return 1;
  // }
}
