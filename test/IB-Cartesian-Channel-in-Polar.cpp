#include <charconv>
#include <numbers>

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

// = Setup =========================================================================================
using Float               = double;
constexpr Float pi        = std::numbers::pi_v<Float>;

constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;
constexpr Float r_min     = 0.5;
constexpr Float r_max     = 2.0;
constexpr Float h         = 1.0;

constexpr Float Uavg      = 1.0;

constexpr Float rho       = 1.0;
constexpr Float mu        = 1.0;
constexpr Float CFL       = 0.7;
constexpr Float tend      = 0.5;

constexpr Float dt_write  = tend / 100.0;

constexpr Rect<Float> wall_bottom{
    .x = -r_max,
    .y = 0.0,
    .w = 2.0 * r_max,
    .h = r_min,
};
constexpr Rect<Float> wall_top{
    .x = -r_max,
    .y = r_min + h,
    .w = 2.0 * r_max,
    .h = r_max - r_min - h,
};
// = Setup =========================================================================================

namespace Expected {

constexpr std::array ns  = {16, 32, 64, 128};
constexpr std::array L1s = {3.17841299e-03, 7.99785696e-04, 1.99373866e-04, 4.89090515e-05};
static_assert(ns.size() == L1s.size());
constexpr auto L1(Index n) { return interp_n2(ns, L1s, n); }

}  // namespace Expected

// =================================================================================================
template <typename Float>
constexpr auto u_analytical(Float y) -> Float {
  constexpr auto y_min = r_min;
  if (y < y_min || y > y_min + h) { return 0.0; }
  const Float s = (y - y_min) / h;
  return Uavg * 6.0 * s * (1.0 - s);
}

// =================================================================================================
// template <typename Float, Layout LAYOUT>
// constexpr void custom_velocity_top_boundary(const Grid<Float, LAYOUT>& grid,
//                                             FaceVector<Float, LAYOUT> u) {
//
//   // - U -----------
//   grid.foreach_range(
//       -u.x.nghost(), u.x.nx() / 2, 0, 1, FOREACH_FUNC {
//         for (j = u.x.ny(); j < u.x.ny() + u.x.nghost(); ++j) {
//           u.x(i, j) = u.x(i, 2 * u.x.ny() - j - 1);
//         }
//       });
//
//   // - V -----------
//   grid.foreach_range(
//       -u.y.nghost(), u.y.nx() / 2, 0, 1, FOREACH_FUNC {
//         for (j = u.y.ny(); j < u.y.ny() + u.y.nghost(); ++j) {
//           u.y(i, j) = u.y(i, 2 * u.y.ny() - j - 2);
//         }
//         u.y(i, u.y.ny() - 1) = (4.0 * u.y(i, u.y.ny() - 2) - u.y(i, u.y.ny() - 3)) / 3.0;
//       });
// }

// =================================================================================================
// template <typename Float, Layout LAYOUT>
// void correct_outflow(const Grid<Float, LAYOUT>& grid, FaceVector<Float, LAYOUT> u) {
//   using Metric     = Metric::Polar;
//
//   const Index jtop = u.y.ny() - 1;
//   Float Q_in       = 0.0;
//   for (Index i = u.y.nx() / 2; i < u.y.nx(); ++i) {
//     const auto A_top  = Metric::H(grid.xm(i), grid.y(jtop)) / Metric::h2(grid.xm(i),
//     grid.y(jtop)); Q_in             += u.y(i, jtop) * A_top;
//   }
//
//   Float Q_out = 0.0;
//   Float A_out = 0.0;
//   for (Index i = 0; i < u.y.nx() / 2; ++i) {
//     const auto A_top  = Metric::H(grid.xm(i), grid.y(jtop)) / Metric::h2(grid.xm(i),
//     grid.y(jtop)); Q_out            += u.y(i, jtop) * A_top; A_out            += A_top;
//   }
//
//   const Float corr = (Q_in - Q_out) / A_out;
//   for (Index i = 0; i < u.y.nx() / 2; ++i) {
//     u.y(i, jtop) += corr;
//   }
// }

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

  const auto output_dir = "./test/output/IB-Cartesian-Channel-in-Polar-"
#if SEMI_ANALYTICAL
                          "SA-"
#else
                          "IE-"
#endif
                          + std::to_string(N) + '/';
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(theta_min, theta_max, 2 * N, r_min, r_max, N, 1, Coordinates::POLAR);

  auto u_old   = grid.alloc_face_vector();
  auto u       = grid.alloc_face_vector();
  auto ui      = grid.alloc_vector();

  auto FUX     = grid.alloc_scalar();
  auto FUY     = grid.alloc_vertex_scalar();
  auto FVX     = grid.alloc_vertex_scalar();
  auto FVY     = grid.alloc_scalar();

  auto div     = grid.alloc_scalar();
  auto p       = grid.alloc_scalar();
  auto dp      = grid.alloc_scalar();

  auto ib_wall = grid.alloc_scalar();
  auto ib_corr = grid.alloc_face_vector();

  Float dt     = 0.0;
  Float t      = 0.0;

  calc_ib_volume_fraction(
      grid,
      [](const Vec2<Float>& xy) {
        return static_cast<Float>(wall_bottom.contains(xy) || wall_top.contains(xy));
      },
      ib_wall);
  calc_ib_correction_shape(grid, wall_bottom, ib_corr);
  calc_ib_correction_shape(grid, wall_top, ib_corr);

  const BConds<Float> u_bconds{
      .left   = Dirichlet<Float>{.val = 0.0},
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top =
          Dirichlet<Float>{
              .val =
                  [](Float theta, Float /*t*/) {
                    const auto p = polar2cartesian(theta, r_max);
                    return -u_analytical(p.y) * std::sin(theta);
                  },
          },
  };

  const BConds<Float> v_bconds{
      .left   = Dirichlet<Float>{.val = 0.0},
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top =
          Dirichlet<Float>{
              .val =
                  [](Float theta, Float /*t*/) {
                    const auto p = polar2cartesian(theta, r_max);
                    return u_analytical(p.y) * std::cos(theta);
                  },
          },
  };

  const BConds<Float> dp_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };

  MultigridSolver solver(grid, dp_bconds);
  Index mg_cycles   = 0;
  Index mg_num_pre  = solver.num_iter_pre();
  Index mg_num_post = solver.num_iter_post();
  Float mg_res      = 0.0;

  // Initial condition: the analytic fully-developed profile.
  grid.foreach_face_i<Dimension::X>(FOREACH_FUNC {
    // const auto theta = grid.theta(i);
    // const auto r     = grid.rm(j);
    // const auto p     = polar2cartesian(theta, r);
    // u.x(i, j)        = -u_analytical(p.y) * std::sin(theta);
    u.x(i, j) = 0.0;
  });
  grid.foreach_face_i<Dimension::Y>(FOREACH_FUNC {
    // const auto theta = grid.thetam(i);
    // const auto r     = grid.r(j);
    // const auto p     = polar2cartesian(theta, r);
    // u.y(i, j)        = u_analytical(p.y) * std::cos(theta);
    u.y(i, j) = 0.0;
  });
  apply_velocity_bconds(grid, u_bconds, v_bconds, u);
  // custom_velocity_top_boundary(grid, u);
  interpolate(grid, u, ui);

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("div", div);
  writer.add_field("wall", ib_wall);
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
  monitor.add_variable(&u_stats.max, "max(u)");
  monitor.add_variable(&v_stats.max, "max(v)");
  monitor.add_variable(&div_max, "absmax(div)");
  monitor.add_variable(&mg_res, "res(MG)");
  monitor.add_variable(&mg_num_pre, "iter_pre(MG)");
  monitor.add_variable(&mg_num_post, "iter_post(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();

  bool any_failed = false;
  IGOR_TIME_SCOPE("Solver")
  while (t < tend && !any_failed) {
    dt = std::min({
        adjust_dt(grid, u, rho, mu, CFL),
        dt_write,
        tend - t,
    });

    copy(u, u_old);

    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      // 1) Predictor
      calc_mom_flux(grid, u, p, rho, mu, FUX, FUY, FVX, FVY);
#if SEMI_ANALYTICAL
      update_u_ib_semi_analytical(
          grid, local_dt, None{}, None{}, FUX, FUY, FVX, FVY, None{}, mu, rho, ib_corr, u_old, u);
#else
      update_u(grid, local_dt, FUX, FUY, FVX, FVY, u_old, u);
      correct_velocity_ib_implicit_euler(grid, ib_corr, rho, mu, local_dt, u);
#endif
      apply_velocity_bconds(grid, u_bconds, v_bconds, u);
      // custom_velocity_top_boundary(grid, u);
      // correct_outflow(grid, u);

      // 2) Pressure correction
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rho / local_dt; });
      if (!solver.solve(dp, div, 1e-3 / local_dt)) {
        Igor::Error("Multigrid solver did not converge after {} cycles: res = {:.8e}",
                    solver.num_cycles(),
                    solver.res());
        any_failed = true;
      }
      mg_num_pre  = solver.num_iter_pre();
      mg_num_post = solver.num_iter_post();
      mg_cycles   = solver.num_cycles();
      mg_res      = solver.res();
      apply_bconds(grid, dp_bconds, dp, t);

      // 3) Project
      correct_velocity(grid, dp, rho, local_dt, u, p);
    }

    interpolate(grid, u, ui);
    calc_div(grid, u, div);

    p_stats   = stats(grid, p);
    u_stats   = stats(grid, u.x);
    v_stats   = stats(grid, u.y);
    div_stats = stats(grid, div);
    div_max   = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

    if (std::isnan(p_stats.max)) {
      Igor::Error("NaN value in p: Simulation diverged.");
      any_failed = true;
    }
    if (std::isnan(u_stats.max)) {
      Igor::Error("NaN value in u: Simulation diverged.");
      any_failed = true;
    }
    if (std::isnan(v_stats.max)) {
      Igor::Error("NaN value in v: Simulation diverged.");
      any_failed = true;
    }
    if (std::isnan(div_max)) {
      Igor::Error("NaN value in div: Simulation diverged.");
      any_failed = true;
    };

    t += dt;
    if (should_save(t, dt, dt_write, tend)) {
      if (!writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  const auto L1 = grid.transform_reduce_range(
      u.x.nx() / 2,
      u.x.nx() / 2 + 1,
      0,
      u.x.ny(),
      Vec2<Float>{},
      FOREACH_FUNC->Vec2<Float> {
        const auto theta  = grid.thetam(i);
        const auto r      = grid.rm(j);
        const auto [x, y] = polar2cartesian(theta, r);

        const auto uthi   = (u.left(i, j) + u.right(i, j)) / 2.0;
        const auto uri    = (u.bottom(i, j) + u.top(i, j)) / 2.0;

        const auto ui     = std::cos(theta) * uri - std::sin(theta) * uthi;
        const auto vi     = std::sin(theta) * uri + std::cos(theta) * uthi;

        const auto u_exp  = u_analytical(y);
        const auto v_exp  = 0.0;
        return {
            .x = std::abs(u_exp - ui) * grid.dr(),
            .y = std::abs(v_exp - vi) * grid.dr(),
        };
      },
      std::plus<>{});

  Igor::Info("L1-u({}) = {:.12e}", N, L1.x);
  Igor::Info("L1-v({}) = {:.12e}", N, L1.y);
  // Igor::Info("L1_exp({}) = {:.8e}", N, Expected::L1(N));
  // if (L1 > 1.1 * Expected::L1(N)) {
  //   Igor::Error("u_theta error does not match expected value: expected {:.8e} but got {:.8e}",
  //               Expected::L1(N),
  //               L1);
  //   any_failed = true;
  // }

  return any_failed ? 1 : 0;
}
