#include <charconv>
#include <numbers>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "BoundaryConditions.hpp"
#include "Common.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"
#include "Mac.hpp"
#include "Monitor.hpp"
#include "MultigridPoisson.hpp"

#include "Test-Common.hpp"

using Float               = double;
constexpr Float pi        = std::numbers::pi_v<Float>;

constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;
constexpr Float r_min     = 1.0;
constexpr Float r_max     = 2.0;

constexpr Float rho       = 1.0;
constexpr Float mu        = 1e-2;
constexpr Float a         = 2.0;
constexpr Float U         = 1.0;

constexpr Float CFL       = 0.5;
constexpr Float tend      = 20.0;
constexpr Float dt_write  = tend / 100.0;

// =================================================================================================
// Hill's spherical vortex
constexpr auto ur(Float theta, Float r) -> Float {
  return -3.0 * U / 2.0 * (1.0 - Igor::sqr(r) / Igor::sqr(a)) * std::cos(theta);
}

constexpr auto utheta(Float theta, Float r) -> Float {
  return 3.0 * U / (2.0 * Igor::sqr(a)) * (Igor::sqr(a) - 2.0 * Igor::sqr(r)) * std::sin(theta);
}

constexpr Float C = -3.0 * U / (4.0 * Igor::sqr(a));
constexpr auto psi(Float theta, Float r) -> Float {
  return C * Igor::sqr(r) * Igor::sqr(std::sin(theta)) * (Igor::sqr(a) - Igor::sqr(r));
}

constexpr auto pressure(Float theta, Float r) -> Float {
  return -rho * ((Igor::sqr(ur(theta, r)) + Igor::sqr(utheta(theta, r))) / 2.0 +
                 10.0 * C * psi(theta, r)) -
         20.0 * mu * C * r * std::cos(theta);
}

// =================================================================================================
namespace Expected {
constexpr std::array ns        = {32, 64, 128};
constexpr std::array L1uthetas = {3.32229590e-03, 8.28386640e-04, 2.04765805e-04};
static_assert(ns.size() == L1uthetas.size());
constexpr std::array L1urs = {1.36022995e-03, 3.32624440e-04, 8.20466134e-05};
static_assert(ns.size() == L1urs.size());
constexpr std::array L1ps = {1.89309975e-03, 4.73727611e-04, 1.17631084e-04};
static_assert(ns.size() == L1ps.size());

constexpr auto L1utheta(Index n) { return interp_n2(ns, L1uthetas, n); }
constexpr auto L1ur(Index n) { return interp_n2(ns, L1urs, n); }
constexpr auto L1p(Index n) { return interp_n2(ns, L1ps, n); }
}  // namespace Expected
// =================================================================================================

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

  const std::string output_dir = "./test/output/Hill-Vortex-" + std::to_string(N) + '/';
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float, Layout::C> grid(theta_min,
                              theta_max,
                              N > 16 ? N : 2 * N,
                              r_min,
                              r_max,
                              N,
                              1,
                              Coordinates::SYMMETRIC_SPHERICAL);

  auto u_old  = grid.alloc_face_vector();
  auto u      = grid.alloc_face_vector();
  auto u_exp  = grid.alloc_face_vector();

  auto FUX    = grid.alloc_scalar();
  auto FUY    = grid.alloc_vertex_scalar();
  auto FVX    = grid.alloc_vertex_scalar();
  auto FVY    = grid.alloc_scalar();
  auto FWZ    = grid.alloc_scalar();

  auto ui     = grid.alloc_vector();
  auto ui_exp = grid.alloc_vector();

  auto p      = grid.alloc_scalar();
  auto dp     = grid.alloc_scalar();
  auto div    = grid.alloc_scalar();

  Float dt    = 0.0;
  Float t     = 0.0;

  // Axis: u_theta = 0 and du_r/dtheta = 0
  const BConds<Float> utheta_bconds{
      .left  = Dirichlet<Float>{.val = 0.0},
      .right = Dirichlet<Float>{.val = 0.0},
      .bottom =
          Dirichlet<Float>{
              .val = [](Float theta, Float /*t*/) { return utheta(theta, r_min); },
          },
      .top =
          Dirichlet<Float>{
              .val = [](Float theta, Float /*t*/) { return utheta(theta, r_max); },
          },
  };
  const BConds<Float> ur_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = [](Float theta, Float /*t*/) { return ur(theta, r_min); }},
      .top    = Dirichlet<Float>{.val = [](Float theta, Float /*t*/) { return ur(theta, r_max); }},
  };

  const BConds<Float> dp_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };

  MultigridSolver solver(grid, dp_bconds);
  Float mg_res    = 0.0;
  Index mg_cycles = 0;

  fill(u, 0.0);
  apply_velocity_bconds(grid, utheta_bconds, ur_bconds, u);
  interpolate(grid, u, ui);

  grid.foreach_face_i<Dimension::X>(
      FOREACH_FUNC { u_exp.x(i, j) = utheta(grid.theta(i), grid.rm(j)); });
  grid.foreach_face_i<Dimension::Y>(
      FOREACH_FUNC { u_exp.y(i, j) = ur(grid.thetam(i), grid.r(j)); });
  apply_velocity_bconds(grid, utheta_bconds, ur_bconds, u_exp);
  interpolate(grid, u_exp, ui_exp);

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("u_exp", ui_exp);
  writer.add_field("p", p);
  writer.add_field("div", div);
  if (!writer.write(t)) { return 1; }

  Stats p_stats   = stats(grid, p);
  Stats u_stats   = stats(grid, u.x);
  Stats v_stats   = stats(grid, u.y);
  Stats div_stats = stats(grid, div);
  Float div_max   = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.min, "min(p)");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.min, "min(u)");
  monitor.add_variable(&u_stats.max, "max(u)");
  monitor.add_variable(&v_stats.min, "min(v)");
  monitor.add_variable(&v_stats.max, "max(v)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.add_variable(&mg_res, "residual(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();

  bool any_failed = false;
  IGOR_TIME_SCOPE("Hill-Vortex-" + std::to_string(N))
  while (t < tend && !any_failed) {
    dt = adjust_dt(grid, u, rho, mu, CFL);
    dt = std::min({dt, dt_write, tend - t});

    copy(u, u_old);

    mg_cycles = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      // 1) Predictor
      calc_mom_flux(grid, u, p, rho, mu, FUX, FUY, FVX, FVY, FWZ);
      update_u(grid, local_dt, FUX, FUY, FVX, FVY, FWZ, u_old, u);
      apply_velocity_bconds(grid, utheta_bconds, ur_bconds, u);

      // 2) Pressure correction
      // A loose tolerance leaves dp unsolved once |div| is small; p then never converges to the
      // steady pressure even though the velocity does.
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rho / local_dt; });
      if (!solver.solve(dp, div, 1e-8 / local_dt)) {
        Igor::Warn("Multigrid solver did not converge after {} cycles: res = {:.8e}",
                   solver.num_cycles(),
                   solver.res());
        any_failed = true;
      }
      mg_res     = solver.res();
      mg_cycles += solver.num_cycles();
      apply_bconds(grid, dp_bconds, dp, t);

      // 3) Project
      correct_velocity(grid, dp, rho, local_dt, u, p);
    }

    interpolate(grid, u, ui);
    calc_div(grid, u, div);

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

  Float L1_utheta = grid.transform_reduce_face_i<Dimension::X>(
      0.0,
      FOREACH_FUNC {
        const auto utheta_exp = utheta(grid.theta(i), grid.rm(j));
        // grid.dv is cell-centered and becomes negative at i = nx (theta > pi); use face volume
        const auto dv_face =
            Metric::SymmetricSpherical::H(grid.theta(i), grid.rm(j)) * grid.dx() * grid.dy();
        return std::abs(utheta_exp - u.x(i, j)) * dv_face;
      },
      std::plus<>{});
  Float L1_ur = grid.transform_reduce_face_i<Dimension::Y>(
      0.0,
      FOREACH_FUNC {
        const auto ur_exp = ur(grid.thetam(i), grid.r(j));
        const auto dv_face =
            Metric::SymmetricSpherical::H(grid.thetam(i), grid.r(j)) * grid.dx() * grid.dy();
        return std::abs(ur_exp - u.y(i, j)) * dv_face;
      },
      std::plus<>{});

  // Pressure is only defined up to a constant; compare the mean-free fields
  auto p_exp = grid.alloc_scalar();
  grid.foreach_i(FOREACH_FUNC { p_exp(i, j) = pressure(grid.thetam(i), grid.rm(j)); });
  const auto p_exp_stats = stats(grid, p_exp);
  p_stats                = stats(grid, p);
  const Float p_offset   = p_stats.sum / p_stats.volume - p_exp_stats.sum / p_exp_stats.volume;
  const Float L1_p       = grid.transform_reduce_i(
      0.0,
      FOREACH_FUNC { return std::abs(p_exp(i, j) - (p(i, j) - p_offset)) * grid.dv(i, j); },
      std::plus<>{});

  Igor::Info("L1_p          = {:.8e}", L1_p);
  Igor::Info("L1_utheta     = {:.8e}", L1_utheta);
  Igor::Info("L1_ur         = {:.8e}", L1_ur);
  Igor::Info("L1_utheta_exp = {:.8e}", Expected::L1utheta(N));
  Igor::Info("L1_ur_exp     = {:.8e}", Expected::L1ur(N));
  Igor::Info("L1_p_exp      = {:.8e}", Expected::L1p(N));

  if (L1_utheta > 1.1 * Expected::L1utheta(N) || std::isnan(L1_utheta)) {
    Igor::Error("u_theta error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1utheta(N),
                L1_utheta);
    any_failed = true;
  }
  if (L1_ur > 1.1 * Expected::L1ur(N) || std::isnan(L1_ur)) {
    Igor::Error("u_r error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1ur(N),
                L1_ur);
    any_failed = true;
  }

  if (L1_p > 1.1 * Expected::L1p(N) || std::isnan(L1_p)) {
    Igor::Error("p error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1p(N),
                L1_p);
    any_failed = true;
  }

  return any_failed ? 1 : 0;
}
