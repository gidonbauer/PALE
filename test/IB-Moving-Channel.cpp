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
// TODO: IB cannot be aligned with the domain boundaries, therefore the offset of 0.5
constexpr Float y_min = -0.5;
constexpr Float y_max = 2.5;

// TODO: The test fails to converge with 2nd order when the Reynolds number is too large (~100)
constexpr Float rho  = 1.0;
constexpr Float mu   = 1e-3;  // 1.0;
constexpr Float Uavg = 1e-2;  // 1.0;
constexpr Vec2<Float> u_wall{.x = 0.0, .y = 1.0};

constexpr Float CFL            = 0.5;
constexpr Float tend           = 1.0;
constexpr Float dt_write       = tend / 100.0;

constexpr Float channel_height = 1.0;
HalfPlane<Float> bottom_wall{.x = 0.0, .y = 0.0, .nx = 0.0, .ny = -1.0};         // NOLINT
HalfPlane<Float> top_wall{.x = 0.0, .y = channel_height, .nx = 0.0, .ny = 1.0};  // NOLINT

constexpr Float Re = rho * channel_height * Uavg / mu;

// =================================================================================================
namespace Expected {

constexpr std::array ns   = {16, 32, 64, 128, 256};
constexpr std::array L1us = {
    1.461238303016e-02,
    4.461331654737e-03,
    1.051345570351e-03,
    2.460960876190e-04,
    6.211731994631e-05,
};
static_assert(ns.size() == L1us.size());
constexpr std::array L1vs = {
    2.119044483303e-03,
    5.816160966582e-04,
    1.224441240651e-04,
    2.984549587223e-05,
    7.374669916040e-06,
};
static_assert(ns.size() == L1vs.size());

constexpr auto L1u(Index n) { return interp_n2(ns, L1us, n); }
constexpr auto L1v(Index n) { return interp_n2(ns, L1vs, n); }

}  // namespace Expected

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
  auto is_solid = [](Float x, Float y) {
    return bottom_wall.contains(x, y) || top_wall.contains(x, y);
  };

  // Rescale the outflow so that it matches the inflow exactly (global continuity).
  Float Qin  = 0.0;
  Float Qout = 0.0;
  for (Index j = 0; j < u.x.ny(); ++j) {
    if (!is_solid(grid.x(0), grid.ym(j))) { Qin += u.x(0, j) * grid.dy(); }
    if (!is_solid(grid.x(u.x.nx() - 1), grid.ym(j))) { Qout += u.x(u.x.nx() - 1, j) * grid.dy(); }
  }
  const Float corr = (Qin - Qout) / (top_wall.y - bottom_wall.y);
  for (Index j = 0; j < u.x.ny(); ++j) {
    if (!is_solid(grid.x(u.x.nx() - 1), grid.ym(j))) { u.x(u.x.nx() - 1, j) += corr; }
  }
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void fill_u_exp(const Grid<Float, LAYOUT>& grid, FaceVector<Float, LAYOUT> u_exp) {
  grid.template foreach_face_a<Dimension::X>(
      FOREACH_FUNC { u_exp.x(i, j) = u_analytical(grid.ym(j)); });
  grid.template foreach_face_a<Dimension::Y>(FOREACH_FUNC { u_exp.y(i, j) = u_wall.y; });
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr auto calc_L1(const Grid<Float, LAYOUT>& grid,
                       const FaceVector<Float, LAYOUT> u,
                       const FaceVector<Float, LAYOUT> u_exp) -> Vec2<Float> {
  const auto L1_u = grid.transform_reduce_range(
      u.x.nx() / 2,
      u.x.nx() / 2 + 1,
      0,
      u.x.ny(),
      Float{0},
      FOREACH_FUNC->Float {
        const auto x = grid.x(i);
        const auto y = grid.ym(j);
        if (bottom_wall.contains(x, y) || top_wall.contains(x, y)) { return 0.0; }
        return std::abs(u_exp.x(i, j) - u.x(i, j)) * grid.dy();
      },
      std::plus<>{});

  const auto L1_v = grid.transform_reduce_range(
      u.y.nx() / 2,
      u.y.nx() / 2 + 1,
      0,
      u.y.ny(),
      Float{0},
      FOREACH_FUNC->Float {
        const auto x = grid.xm(i);
        const auto y = grid.y(j);
        if (bottom_wall.contains(x, y) || top_wall.contains(x, y)) { return 0.0; }
        return std::abs(u_exp.y(i, j) - u.y(i, j)) * grid.dy();
      },
      std::plus<>{});

  return Vec2<Float>{.x = L1_u, .y = L1_v};
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

  Igor::Info("Re = {}", Re);

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

  auto u_exp       = grid.alloc_face_vector();
  auto ui_exp      = grid.alloc_vector();
  fill_u_exp(grid, u_exp);
  interpolate(grid, u_exp, ui_exp);

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
      .bottom = Dirichlet<Float>{.val = u_wall.y},
      .top    = Dirichlet<Float>{.val = u_wall.y},
  };
  fill_u_exp(grid, u);
  interpolate(grid, u, ui);

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("div", div);
  writer.add_field("ua", ui_exp);
  writer.add_field("psi", ib_levelset);
  if (!writer.write(t)) { return 1; }

  Stats p_stats   = stats(grid, p);
  Stats u_stats   = stats(grid, u.x);
  Stats v_stats   = stats(grid, u.y);
  Stats div_stats = stats(grid, div);
  Float div_max   = std::max(std::abs(div_stats.min), std::abs(div_stats.max));
  Vec2<Float> L1  = calc_L1(grid, u, u_exp);

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.min, "min(u)");
  monitor.add_variable(&u_stats.max, "max(u)");
  monitor.add_variable(&v_stats.min, "min(v)");
  monitor.add_variable(&v_stats.max, "max(v)");
  monitor.add_variable(&L1.x, "L1(u)");
  monitor.add_variable(&L1.y, "L1(v)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.add_variable(&mg_res, "residual(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();

  IGOR_TIME_SCOPE("Solver")
  while (tend - t > 1e-8) {
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
#if 1
      update_u(grid, local_dt, FUX, FUY, FVX, FVY, u_old, u);
      correct_velocity_ib_implicit_euler(grid, ib_corr, rho, mu, local_dt, u_wall, u);
#else
      // TODO: This does not work at the moment; Near the outlet, we get very high velocities in
      //       some time steps.
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
                                  ib_corr,
                                  u_wall,
                                  u_old,
                                  u);
#endif
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

    fill_u_exp(grid, u_exp);
    interpolate(grid, u_exp, ui_exp);

    L1         = calc_L1(grid, u, u_exp);

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

  Igor::Info("L1_u({})     = {:.12e}", N, L1.x);
  Igor::Info("L1_u_exp({}) = {:.12e}", N, Expected::L1u(N));
  Igor::Info("L1_v({})     = {:.12e}", N, L1.y);
  Igor::Info("L1_v_exp({}) = {:.12e}", N, Expected::L1v(N));

  bool any_failed = false;
  if (L1.x > 1.1 * Expected::L1u(N) || std::isnan(L1.x)) {
    Igor::Error("u error does not match expected value: expected {:.12e} but got {:.12e}",
                Expected::L1u(N),
                L1.x);
    any_failed = true;
  }
  if (L1.y > 1.1 * Expected::L1v(N) || std::isnan(L1.y)) {
    Igor::Error("v error does not match expected value: expected {:.12e} but got {:.12e}",
                Expected::L1v(N),
                L1.y);
    any_failed = true;
  }
  return any_failed ? 1 : 0;
}
