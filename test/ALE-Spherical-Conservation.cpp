#include <charconv>
#include <numbers>

#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "ALE.hpp"
#include "Advection-Diffusion.hpp"
#include "BoundaryConditions.hpp"
#include "Common.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"
#include "Mac.hpp"
#include "Monitor.hpp"
#include "MultigridPoisson.hpp"

using Float               = double;

constexpr Float pi        = std::numbers::pi_v<Float>;

constexpr Float r_min     = 1.0;
constexpr Float r_max     = 10.0;
constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;

constexpr Float rho       = 1.0;
constexpr Float mu        = 1e-3;
constexpr Float D         = 1e-1;

constexpr Float CFL       = 0.5;
constexpr Float tend      = 1.0;
constexpr Float dt_write  = tend / 100.0;

constexpr Vec2<Float> w{.x = 0.0, .y = 1.0};

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void correct_outflow(const Grid<Float, LAYOUT>& grid, FaceVector<Float, LAYOUT> u) {
  // Rescale the outflow so that it matches the inflow exactly (global continuity).
  Float Qin  = 0.0;
  Float Qout = 0.0;
  for (Index i = 0; i < u.y.nx(); ++i) {
    Qin  += u.y(i, 0) * grid.y_min() * grid.dx();
    Qout += u.y(i, u.y.ny() - 1) * grid.y_max() * grid.dx();
  }
  const Float corr = (Qin - Qout) / (static_cast<Float>(u.y.nx()) * grid.y_max() * grid.dx());
  for (Index i = 0; i < u.y.nx(); ++i) {
    u.y(i, u.y.ny() - 1) += corr;
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

  const auto output_dir = "test/output/ALE-Spherical-Conservation-" + std::to_string(N);
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(theta_min, theta_max, N, r_min, r_max, N, 3, Coordinates::SYMMETRIC_SPHERICAL);
  auto u_old = grid.alloc_face_vector();
  auto u     = grid.alloc_face_vector();
  auto ui    = grid.alloc_vector();

  auto FUX   = grid.alloc_scalar();
  auto FUY   = grid.alloc_vertex_scalar();
  auto FVX   = grid.alloc_vertex_scalar();
  auto FVY   = grid.alloc_scalar();
  auto FWZ   = grid.alloc_scalar();

  auto div   = grid.alloc_scalar();
  auto p     = grid.alloc_scalar();
  auto dp    = grid.alloc_scalar();

  auto J_old = grid.alloc_scalar();
  auto J     = grid.alloc_scalar();
  calc_H(grid, J);

  auto s_old = grid.alloc_scalar();
  auto s     = grid.alloc_scalar();
  auto Fs    = grid.alloc_face_vector();

  Float t    = 0.0;
  Float dt   = 1e-1;

  const BConds<Float> uth_bconds{
      .left   = Dirichlet<Float>{.val = 0.0},
      .right  = Dirichlet<Float>{.val = 0.0},
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Neumann{.clipped = false},
  };
  const BConds<Float> ur_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = w.r()},
      .top    = Neumann{.clipped = false},
  };
  fill(u.x, 0.0);
  fill(u.y, 0.0);
  apply_velocity_bconds(grid, uth_bconds, ur_bconds, u);
  interpolate(grid, u, ui);

  const BConds<Float> dp_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };
  MultigridSolver solver(grid, dp_bconds);

  const BConds<Float> s_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };
  grid.foreach_i(FOREACH_FUNC { s(i, j) = grid.r(j) < 2.0 ? 1.0 : 0.0; });
  apply_bconds(grid, s_bconds, s, t);

  // - Output ------------------------------------------------------------------
  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("div", div);
  writer.add_field("s", s);
  writer.add_field("J", J);
  if (!writer.write(t)) { return 1; }

  Stats p_stats      = stats(grid, p);
  Stats u_stats      = stats(grid, u.x);
  Stats v_stats      = stats(grid, u.y);
  Stats div_stats    = stats(grid, div);
  Stats s_stats      = stats(grid, s);
  Float div_max      = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

  const Float s0_sum = s_stats.sum;

  Float mg_res       = 0.0;
  Index mg_cycles    = 0;

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.max, "max(u_theta)");
  monitor.add_variable(&v_stats.max, "max(u_r)");
  monitor.add_variable(&s_stats.min, "min(s)");
  monitor.add_variable(&s_stats.max, "max(s)");
  monitor.add_variable(&s_stats.sum, "sum(s)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.add_variable(&mg_res, "res(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();
  // - Output ------------------------------------------------------------------

  bool any_failed = false;
  IGOR_TIME_SCOPE("ALE-Spherical-Conservation-" + std::to_string(N))
  while (t < tend && !any_failed) {
    dt = std::min({
        adjust_dt(grid, u, rho, mu, CFL),
        advection_adjust_dt(grid, D, CFL),
        ale_adjust_dt(grid, w, CFL),
        dt_write,
        std::max(tend - t, 1e-6),
    });

    copy(u, u_old);
    copy(s, s_old);
    copy(J, J_old);

    mg_cycles = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? 0.5 * dt : dt;

      // 1) Update the cell volume metric J
      update_J(grid, local_dt, w, J_old, J);

      // 2) Prediction
      calc_mom_flux(grid, u, p, rho, mu, w, FUX, FUY, FVX, FVY, FWZ);
      update_u(grid, local_dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, u_old, u);
      apply_velocity_bconds(grid, uth_bconds, ur_bconds, u);
      correct_outflow(grid, u);

      // 3) Pressure calculation
      grid.move_grid_by_velocity(w, 0.5 * dt);
      solver.move_grid_by_velocity(w, 0.5 * dt);
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rho / local_dt; });
      if (!solver.solve(dp, div, 1e-6 / local_dt)) {
        Igor::Error("t={:.8f}: Multigrid solver did not converge after {} cycles: res = {:.8e}",
                    t,
                    solver.num_cycles(),
                    solver.res());
        any_failed = true;
      }
      mg_res     = solver.res();
      mg_cycles += solver.num_cycles();
      apply_bconds(grid, dp_bconds, dp, t);

      // 4) Projection
      correct_velocity(grid, dp, rho, local_dt, u, p);
      apply_velocity_bconds_only_periodic(grid, uth_bconds, ur_bconds, u);
      grid.move_grid_by_velocity(w, -0.5 * dt);

      // 5) Update scalar
      calc_advection_flux(grid, u, s, w, D, Fs);
      update_s(grid, local_dt, J_old, J, Fs, s_old, s);
      apply_bconds(grid, s_bconds, s, t);

      // 6) Update the physical position of the grid
      grid.move_grid_by_velocity(w, 0.5 * dt);
      // solver.move_grid_by_velocity(w, 0.5 * dt);
    }
    calc_div(grid, u, div);
    interpolate(grid, u, ui);

    p_stats    = stats(grid, p);
    u_stats    = stats(grid, u.x);
    v_stats    = stats(grid, u.y);
    div_stats  = stats(grid, div);
    s_stats    = stats(grid, s);
    div_max    = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

    t         += dt;
    monitor.write();
    if (should_save(t, dt, dt_write, tend)) {
      writer.update_grid(grid);
      if (!writer.write(t)) { return 1; }
    }
  }

  const auto abserr_conservation = std::abs(s_stats.sum - s0_sum);
  Igor::Info("sum(s0) = {:.12e}", s0_sum);
  Igor::Info("sum(s)  = {:.12e}", s_stats.sum);
  Igor::Info("abs. conservation error = {:.12e}", abserr_conservation);

  const auto tol = [=] {
    if (N <= 16) { return 1e-10; }
    return 1e-12;
  }();
  if (abserr_conservation > tol || std::isnan(abserr_conservation)) {
    Igor::Error("Did not conserve scalar `s`, abs. error of conservation is {:.16}, expected <{}",
                abserr_conservation,
                tol);
    any_failed = true;
  }

  return any_failed ? 1 : 0;
}
