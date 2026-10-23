#include <charconv>
#include <numbers>

#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#define PALE_ASLAM_WENO
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
#include "MultigridPoisson.hpp"

// = Setup =========================================================================================
using Float               = double;
constexpr Float pi        = std::numbers::pi_v<Float>;

constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;
constexpr Float r_min     = 1.0;
constexpr Float r_max     = 2.0;

constexpr Float Re        = 1e3;                      // [-]
constexpr Float Pe        = 1e2;                      // [-]
constexpr Float rho       = 1.0;                      // [kg/m^3]
constexpr Float mu        = 1e-3;                     // [Pa*s]
constexpr Float Uinf      = Re * mu / (rho * r_min);  // [m/s]
constexpr Float D         = (Uinf * r_min) / Pe;      // [m^2/s]

constexpr Float CFL       = 0.7;
constexpr Float tend      = 50.0;
constexpr Float dt_write  = tend / 100.0;

constexpr HalfPlane<Float> wall{.x = -1.5, .y = 0.0, .nx = -1.0, .ny = 0.0};
constexpr Float h = Igor::sqrt(Igor::sqr(r_max) - Igor::sqr(wall.x));
// = Setup =========================================================================================

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

  const auto output_dir = get_output_directory() + "/" + std::to_string(N);
  if (!init_output_directory(output_dir)) { return 1; }

  Igor::Info("Re   = {}", Re);
  Igor::Info("Pe   = {}", Pe);
  Igor::Info("Uinf = {}", Uinf);
  Igor::Info("D    = {}", D);

  Grid<Float> grid(theta_min, theta_max, N, r_min, r_max, N, 3, Coordinates::POLAR);

  auto u_old             = grid.alloc_face_vector();
  auto u                 = grid.alloc_face_vector();
  auto ui                = grid.alloc_vector();

  auto FUX               = grid.alloc_scalar();
  auto FUY               = grid.alloc_vertex_scalar();
  auto FVX               = grid.alloc_vertex_scalar();
  auto FVY               = grid.alloc_scalar();

  auto div               = grid.alloc_scalar();
  auto p                 = grid.alloc_scalar();
  auto dp                = grid.alloc_scalar();

  auto s_old             = grid.alloc_scalar();
  auto s                 = grid.alloc_scalar();
  auto Fs                = grid.alloc_face_vector();

  auto ib_corr_face      = grid.alloc_face_vector();
  auto ib_levelset       = grid.alloc_scalar();
  auto ib_free           = FOREACH_FUNC { return ib_levelset(i, j) <= 0.0; };

  Float dt               = 0.0;
  Float t                = 0.0;

  Float s_sum_exp        = 0.0;

  const Float dsdn       = 1.0;
  const Float band_width = 3.0;

  calc_ib_correction_shape(grid, wall, ib_corr_face);
  calc_ib_levelset_shape(grid, wall, ib_levelset);

  const BConds<Float> u_bconds{
      .left   = Periodic{},
      .right  = Periodic{},
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Neumann(),
  };
  const BConds<Float> v_bconds{
      .left   = Periodic{},
      .right  = Periodic{},
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Neumann(),
  };

  const BConds<Float> s_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };

  const BConds<Float> dp_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };
  MultigridSolver solver(grid, dp_bconds);
  Index mg_cycles = 0;
  Float mg_res    = 0.0;

  grid.foreach_a(FOREACH_FUNC {
    // s(i, j) = ib_levelset(i, j) <= 0.0 ? ib_levelset(i, j) : 0.0;
    s(i, j) = 0.0;
  });
  apply_bconds(grid, s_bconds, s, t);
  Index aslam_iter = neumann_extrapolate(grid, s, ib_levelset, dsdn, band_width);

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
  monitor.add_variable(&u_stats.max, "max(u_theta)");
  monitor.add_variable(&v_stats.max, "max(u_r)");
  monitor.add_variable(&s_stats.min, "min(s)");
  monitor.add_variable(&s_stats.max, "max(s)");
  monitor.add_variable(&s_stats.sum, "sum(s)");
  monitor.add_variable(&s_sum_exp, "exp. sum(s)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.add_variable(&mg_res, "res(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.add_variable(&aslam_iter, "iter(Aslam)");
  monitor.write();

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

    mg_cycles  = 0;
    aslam_iter = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      // 1) Predictor
      calc_mom_flux(grid, u, p, rho, mu, FUX, FUY, FVX, FVY);
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
      apply_velocity_bconds(grid, u_bconds, v_bconds, u);

      // 2) Pressure correction
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rho / local_dt; });
      if (!solver.solve(dp, div, 1e-4 / local_dt)) {
        Igor::Warn("t={:.8f}: Multigrid solver did not converge after {} cycles: res = {:.8e}",
                   t,
                   solver.num_cycles(),
                   solver.res());
      }
      mg_cycles += solver.num_cycles();
      mg_res     = solver.res();
      apply_bconds(grid, dp_bconds, dp, t);

      // 3) Project
      correct_velocity(grid, dp, rho, local_dt, u, p);
      apply_velocity_bconds_only_periodic(grid, u_bconds, v_bconds, u);

      // 4) Update temperature
      calc_advection_flux(grid, u, s, D, Fs);
      update_s(grid, local_dt, Fs, s_old, s);
      apply_bconds(grid, s_bconds, s, t);
      aslam_iter += neumann_extrapolate(grid, s, ib_levelset, dsdn, band_width);
    }

    interpolate(grid, u, ui);
    calc_div(grid, u, div);

    p_stats    = stats(grid, p);
    u_stats    = stats(grid, u.x);
    v_stats    = stats(grid, u.y);
    s_stats    = stats(grid, s, ib_free);
    div_stats  = stats(grid, div);
    div_max    = std::max(std::abs(div_stats.min), std::abs(div_stats.max));
    s_sum_exp += h * D * dsdn * dt;

    t         += dt;
    if (should_save(t, dt, dt_write, tend)) {
      if (!writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Ok.");
}
