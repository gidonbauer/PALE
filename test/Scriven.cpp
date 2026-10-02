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
#include "Quadrature.hpp"
#include "Test-Common.hpp"

#include "Scriven-Analytical.hpp"

#ifndef DIMENSION
#define DIMENSION 2
#endif  // DIMENSION

#if DIMENSION != 2 && DIMENSION != 3
#error "DIMENSION must be 2 or 3"
#endif

using Float               = double;

constexpr Float pi        = std::numbers::pi_v<Float>;

constexpr Float r_min     = 0.25e-3;  // Initial radius                     [m]
constexpr Float r_max     = 20.0 * r_min;
constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;

// Liquid properties
constexpr Float rhol   = 422.36;                 // Density                   [kg/m^3]
constexpr Float mul    = 1.1677e-4;              // Viscosity                 [Pa*s]
constexpr Float cpl    = 3.4811e3;               // Specific heat capacity    [J/(kg K)]
constexpr Float kappal = 0.18370;                // Thermal conductivity      [W/(m K)]
constexpr Float alphal = kappal / (rhol * cpl);  // Thermal diffusivity       [m^2/s]

// Gas properties
constexpr Float rhog = 1.8164;  // Density                   [kg/m^3]
// constexpr Float mug    = 4.3241e-06;             // Viscosity                 [Pa*s]
constexpr Float cpg = 2.2177e3;  // Specific heat capacity    [J/(kg K)]
// constexpr Float kappag = 0.011415;               // Thermal conductivity      [W/(m K)]
// constexpr Float alphag = kappag / (rhog * cpg);  // Thermal diffusivity       [m^2/s]

// Temperatures
constexpr Float Tsat = 111.0;     // Saturation temperature    [K]
constexpr Float Tinf = 111.26;    // Bulk temperature          [K]
constexpr Float hev  = 5.1083e5;  // Enthalpy of vaporization  [J/kg]

// Dimensionless numbers relevant for Scriven case
constexpr Float eps = (rhol - rhog) / rhol;  // Relative difference of density [-]
// constexpr Float Ja    = rhol * cpl * (Tinf - Tsat) / (hev * rhog); // Simplified Jakob number [-]
constexpr Float Ja =
    rhol * cpl * (Tinf - Tsat) / (rhog * (hev + (cpl - cpg) * (Tinf - Tsat)));  // Jakob number [-]

constexpr Float CFL   = 0.5;
constexpr Float r_end = 2.0 * r_min;  // Final radius                       [m]

// =================================================================================================
namespace Expected {

#if DIMENSION == 2

constexpr std::array ns    = {16, 32, 64, 128};
constexpr std::array L1s_T = {
    5.340780646253e-06,
    1.225328329217e-06,
    2.777814157317e-07,
    8.614552824374e-08,
};
static_assert(ns.size() == L1s_T.size());
constexpr std::array L1s_r = {
    1.811738822040e-05,
    3.719041159722e-06,
    5.506450944890e-07,
    1.512516540701e-08,
};
static_assert(ns.size() == L1s_r.size());
constexpr std::array L1s_r_dot = {
    2.075259287376e-05,
    4.245176862502e-06,
    6.853033006766e-07,
    4.784288725711e-08,
};
static_assert(ns.size() == L1s_r_dot.size());

#else

constexpr std::array ns    = {16, 32, 64, 128, 256};
constexpr std::array L1s_T = {
    1.115073130461e-05,
    2.845126719225e-06,
    5.844123876562e-07,
    1.378742611328e-07,
    6.244514080612e-08,
};
static_assert(ns.size() == L1s_T.size());
constexpr std::array L1s_r = {
    2.393055356765e-05,
    5.855476183342e-06,
    9.661163921103e-07,
    8.736656038601e-08,
    4.616879138168e-08,
};
static_assert(ns.size() == L1s_r.size());
constexpr std::array L1s_r_dot = {
    5.000687948203e-05,
    1.226306482408e-05,
    2.200808856663e-06,
    2.862390798302e-07,
    7.150750763645e-08,
};
static_assert(ns.size() == L1s_r_dot.size());

#endif

constexpr auto L1_T(Index n) { return interp_n2(ns, L1s_T, n); }
constexpr auto L1_r(Index n) { return interp_n2(ns, L1s_r, n); }
constexpr auto L1_r_dot(Index n) { return interp_n2(ns, L1s_r_dot, n); }

}  // namespace Expected

// =================================================================================================
template <typename Float, Layout LAYOUT>
void correct_outflow(const Grid<Float, LAYOUT>& grid, FaceVector<Float, LAYOUT> u) {
#if DIMENSION == 2
  using Metric = Metric::Polar;
#else
  using Metric = Metric::SymmetricSpherical;
#endif

  const Index jtop = u.y.ny() - 1;
  Float Q_in       = 0.0;
  Float Q_out      = 0.0;
  Float A_out      = 0.0;
  for (Index i = 0; i < u.y.nx(); ++i) {
    const auto A_bot  = Metric::H(grid.xm(i), grid.y(0)) / Metric::h2(grid.xm(i), grid.y(0));
    const auto A_top  = Metric::H(grid.xm(i), grid.y(jtop)) / Metric::h2(grid.xm(i), grid.y(jtop));
    Q_in             += u.y(i, 0) * A_bot;
    Q_out            += u.y(i, jtop) * A_top;
    A_out            += A_top;
  }
  const Float corr = (Q_in - Q_out) / A_out;
  for (Index i = 0; i < u.y.nx(); ++i) {
    u.y(i, jtop) += corr;
  }
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr auto calc_m_dot_total(const Grid<Float, LAYOUT>& grid, Scalar<Float, LAYOUT> T) -> Float {
#if DIMENSION == 2
  const auto m_dot_total = grid.transform_reduce_range(
      0,
      grid.nx(),
      0,
      1,
      0.0,
      [=](Index i, Index /*j*/) -> Float {
        // Second order one-sided finite differences on non-uniform grid
        const auto dTdr  = (-8.0 * Tsat + 9.0 * T(i, 0) - T(i, 1)) / (3.0 * grid.dr());
        const auto m_dot = kappal * dTdr / hev;  // Assume dTdr=0 in the gas phase
        return m_dot;
      },
      std::plus<>{});
  return m_dot_total * 2.0 * grid.dtheta() * grid.r_min();
#else
  const auto m_dot_total = grid.transform_reduce_range(
      0,
      grid.nx(),
      0,
      1,
      0.0,
      [=](Index i, Index /*j*/) -> Float {
        // Second order one-sided finite differences on non-uniform grid
        const auto dTdr  = (-8.0 * Tsat + 9.0 * T(i, 0) - T(i, 1)) / (3.0 * grid.dr());
        const auto m_dot = kappal * dTdr / hev;  // Assume dTdr=0 in the gas phase
        return m_dot * (std::cos(grid.theta(i)) - std::cos(grid.theta(i + 1)));
      },
      std::plus<>{});
  return m_dot_total * 2.0 * pi * Igor::sqr(grid.r_min());
#endif
}

template <typename Float>
constexpr auto calc_r_dot(Float r, Float m_dot_total) -> Float {
#if DIMENSION == 2
  return m_dot_total / (2.0 * pi * r * rhog);
#else
  return m_dot_total / (4.0 * pi * Igor::sqr(r) * rhog);
#endif
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

  const auto output_dir =
      "./test/output/Scriven-" + std::to_string(DIMENSION) + "D-" + std::to_string(N) + "/";
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(theta_min,
                   theta_max,
                   N,
                   r_min,
                   r_max,
                   N,
                   3,
                   DIMENSION == 2 ? Coordinates::POLAR : Coordinates::SYMMETRIC_SPHERICAL);
  auto u_old = grid.alloc_face_vector();
  auto u     = grid.alloc_face_vector();
  auto ui    = grid.alloc_vector();

  auto FUX   = grid.alloc_scalar();
  auto FUY   = grid.alloc_vertex_scalar();
  auto FVX   = grid.alloc_vertex_scalar();
  auto FVY   = grid.alloc_scalar();
#if DIMENSION == 2
  auto FWZ = None{};
#else
  auto FWZ = grid.alloc_scalar();
#endif

  auto div   = grid.alloc_scalar();
  auto p     = grid.alloc_scalar();
  auto dp    = grid.alloc_scalar();

  auto T_old = grid.alloc_scalar();
  auto T     = grid.alloc_scalar();
  auto FT    = grid.alloc_face_vector();

  auto J_old = grid.alloc_scalar();
  auto J     = grid.alloc_scalar();
  calc_H(grid, J);

  Scriven::Params params{
      .Ja        = Ja,
      .eps       = eps,
      .alpha     = alphal,
      .Tsat      = Tsat,
      .Tinf      = Tinf,
      .beta      = 0.0,
      .dimension = DIMENSION,
  };
  Scriven::calc_beta(params);
  Igor::Info("Scriven::Params = {{");
  Igor::Info("  .Ja        = {}", params.Ja);
  Igor::Info("  .eps       = {}", params.eps);
  Igor::Info("  .alpha     = {}", params.alpha);
  Igor::Info("  .Tsat      = {}", params.Tsat);
  Igor::Info("  .Tinf      = {}", params.Tinf);
  Igor::Info("  .beta      = {}", params.beta);
  Igor::Info("  .dimension = {}", params.dimension);
  Igor::Info("}}");

  Float t              = Scriven::t(r_min, params);
  const Float tend     = Scriven::t(r_end, params);
  const Float dt_write = tend / 100.0;
  Float dt             = dt_write;

  Igor::Info("t0   = {}", t);
  Igor::Info("tend = {}", tend);
  Igor::Info("R0   = {}", r_min);
  Igor::Info("Rend = {}", r_end);

  // - Write setup to JSON -------------------------------------------------------------------------
  {
    const auto json_filename = output_dir + "/setup.json";
    std::ofstream json_out(json_filename);
    if (!json_out) {
      Igor::Error("Could not open `{}`: {}", json_filename, std::strerror(errno));
      return 1;
    }

    json_out << "{\n";
    json_out << "  " << R"("Ja": )" << std::setprecision(12) << std::scientific << params.Ja
             << ",\n";
    json_out << "  " << R"("eps": )" << std::setprecision(12) << std::scientific << params.eps
             << ",\n";
    json_out << "  " << R"("alpha": )" << std::setprecision(12) << std::scientific << params.alpha
             << ",\n";
    json_out << "  " << R"("Tsat": )" << std::setprecision(12) << std::scientific << params.Tsat
             << ",\n";
    json_out << "  " << R"("Tinf": )" << std::setprecision(12) << std::scientific << params.Tinf
             << ",\n";
    json_out << "  " << R"("beta": )" << std::setprecision(12) << std::scientific << params.beta
             << ",\n";
    json_out << "  " << R"("dimension": )" << params.dimension << ",\n";
    json_out << "  " << R"("t0": )" << std::setprecision(12) << std::scientific << t << ",\n";
    json_out << "  " << R"("tend": )" << std::setprecision(12) << std::scientific << tend << ",\n";
    json_out << "  " << R"("R0": )" << std::setprecision(12) << std::scientific << r_min << ",\n";
    json_out << "  " << R"("Rend": )" << std::setprecision(12) << std::scientific << r_end << '\n';
    json_out << "}\n";
  }
  // - Write setup to JSON -------------------------------------------------------------------------

  Vec2<Float> w{};

  const BConds<Float> uth_bconds{
      .left   = Dirichlet<Float>{.val = 0.0},
      .right  = Dirichlet<Float>{.val = 0.0},
      .bottom = Dirichlet<Float>{.val = 0.0},
      .top    = Neumann{.clipped = false},
  };
  BConds<Float> ur_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = eps * w.r()},
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

  const BConds<Float> T_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = Tsat},
      .top    = Dirichlet<Float>{.val = Tinf},
  };
  grid.foreach_i(FOREACH_FUNC { T(i, j) = Scriven::T(grid.rm(j), t, params); });
  apply_bconds(grid, T_bconds, T, t);

  // - Output ------------------------------------------------------------------
  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("T", T);
  writer.add_field("div", div);
  if (!writer.write(t)) { return 1; }

  Stats p_stats      = stats(grid, p);
  Stats u_stats      = stats(grid, u.x);
  Stats v_stats      = stats(grid, u.y);
  Stats T_stats      = stats(grid, T);
  Stats div_stats    = stats(grid, div);
  Float div_max      = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

  Float m_dot_total  = calc_m_dot_total(grid, T);
  Float r            = grid.y_min();
  Float r_dot        = calc_r_dot(r, m_dot_total);
  Float beta_eff     = Scriven::beta_eff(r, r_dot, params);

  Float r_abserr     = std::abs(r - Scriven::R(t, params));
  Float r_dot_abserr = std::abs(r_dot - Scriven::R_dot(t, params));
  Float beta_abserr  = std::abs(beta_eff - params.beta);
  Float r_relerr     = r_abserr / Scriven::R(t, params);
  Float r_dot_relerr = r_dot_abserr / Scriven::R_dot(t, params);
  Float beta_relerr  = beta_abserr / params.beta;

  std::vector<Float> rs{};
  std::vector<Float> r_dots{};
  std::vector<Float> ts{};
  rs.push_back(r);
  r_dots.push_back(r_dot);
  ts.push_back(t);

  Float mg_res    = 0.0;
  Index mg_cycles = 0;

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.max, "max(u_theta)");
  monitor.add_variable(&v_stats.max, "max(u_r)");
  monitor.add_variable(&T_stats.min, "min(T)");
  monitor.add_variable(&T_stats.max, "max(T)");
  monitor.add_variable(&m_dot_total, "m_dot_total");
  monitor.add_variable(&r_dot, "r_dot");
  monitor.add_variable(&r, "r");
  monitor.add_variable(&beta_eff, "beta_eff");
  monitor.add_variable(&mg_res, "res(MG)");
  monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();

  Monitor<Float> error_monitor(output_dir + "/error.log");
  error_monitor.add_variable(&r_abserr, "abserr(r)");
  error_monitor.add_variable(&r_dot_abserr, "abserr(r_dot)");
  error_monitor.add_variable(&beta_abserr, "abserr(beta)");
  error_monitor.add_variable(&r_relerr, "relerr(r)");
  error_monitor.add_variable(&r_dot_relerr, "relerr(r_dot)");
  error_monitor.add_variable(&beta_relerr, "relerr(beta)");
  error_monitor.add_variable(&div_max, "absmax(div)");
  error_monitor.write();
  // - Output ------------------------------------------------------------------

  bool any_failed = false;
  IGOR_TIME_SCOPE("Scriven-" + std::to_string(DIMENSION) + "D-" + std::to_string(N))
  while (t < tend && !any_failed) {
    dt = std::min({
        adjust_dt(grid, u, rhol, mul, CFL),
        advection_adjust_dt(grid, alphal, CFL),
        ale_adjust_dt(grid, w, CFL),
        dt_write,
        std::max(tend - t, 1e-6),
    });

    copy(u, u_old);
    copy(T, T_old);
    copy(J, J_old);

    mg_cycles = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? 0.5 * dt : dt;

      // 1) Mass exchange -> grid velocity
      m_dot_total      = calc_m_dot_total(grid, T);
      r_dot            = calc_r_dot(grid.y_min(), m_dot_total);
      w.r()            = r_dot;
      ur_bconds.bottom = Dirichlet<Float>{.val = eps * w.r()};

      // 2) Update J
      update_J(grid, local_dt, w, J_old, J);

      // 3) Prediction
      calc_mom_flux(grid, u, p, rhol, mul, w, FUX, FUY, FVX, FVY, FWZ);
      update_u(grid, local_dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, u_old, u);
      apply_velocity_bconds(grid, uth_bconds, ur_bconds, u);
      correct_outflow(grid, u);

      // 4) Pressure calculation
      // 4.1) Move grid for pressure correction
      grid.move_grid_by_velocity(w, 0.5 * dt);
      solver.move_grid_by_velocity(w, 0.5 * dt);
      // 4.2) Actual correction
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rhol / local_dt; });
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

      // 5) Projection
      // 5.1) Actual projection
      correct_velocity(grid, dp, rhol, local_dt, u, p);
      apply_velocity_bconds_only_periodic(grid, uth_bconds, ur_bconds, u);
      // 5.2) Move grid back for scalar transport
      grid.move_grid_by_velocity(w, -0.5 * dt);

      // 6) Update temperature
      calc_advection_flux(grid, u, T, w, alphal, FT);
      update_s(grid, local_dt, J_old, J, FT, T_old, T);
      apply_bconds(grid, T_bconds, T, t);

      // 7) Update the physical position of the grid
      grid.move_grid_by_velocity(w, 0.5 * dt);
      // solver.move_grid_by_velocity(w, 0.5 * dt); // Already moved in step 4.1)
    }
    calc_div(grid, u, div);
    interpolate(grid, u, ui);

    p_stats       = stats(grid, p);
    u_stats       = stats(grid, u.x);
    v_stats       = stats(grid, u.y);
    div_stats     = stats(grid, div);
    T_stats       = stats(grid, T);
    div_max       = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

    t            += dt;
    beta_eff      = Scriven::beta_eff(r, r_dot, params);
    r             = grid.y_min();
    m_dot_total   = calc_m_dot_total(grid, T);
    r_dot         = calc_r_dot(grid.y_min(), m_dot_total);
    r_abserr      = std::abs(r - Scriven::R(t, params));
    r_dot_abserr  = std::abs(r_dot - Scriven::R_dot(t, params));
    beta_abserr   = std::abs(beta_eff - params.beta);
    r_relerr      = r_abserr / Scriven::R(t, params);
    r_dot_relerr  = r_dot_abserr / Scriven::R_dot(t, params);
    beta_relerr   = beta_abserr / params.beta;

    rs.push_back(r);
    r_dots.push_back(r_dot);
    ts.push_back(t);

    monitor.write();
    error_monitor.write();
    if (should_save(t, dt, dt_write, tend)) {
      writer.update_grid(grid);
      if (!writer.write(t)) { return 1; }
    }
  }

  const auto L1_T = grid.transform_reduce_range(
      grid.ntheta() / 2,
      grid.ntheta() / 2 + 1,
      0,
      grid.nr(),
      0.0,
      FOREACH_FUNC->Float {
        const auto r     = grid.rm(j);
        const auto T_exp = Scriven::T(r, t, params);
        return std::abs(T(i, j) - T_exp) * grid.dr();
      },
      std::plus<>{});

  for (size_t i = 0; i < ts.size(); ++i) {
    rs[i]     = std::abs(rs[i] - Scriven::R(ts[i], params));
    r_dots[i] = std::abs(r_dots[i] - Scriven::R_dot(ts[i], params));
  }
  const auto L1_r     = simpson(std::span<const Float>{rs}, std::span<const Float>{ts});
  const auto L1_r_dot = simpson(std::span<const Float>{r_dots}, std::span<const Float>{ts});

  std::cout << '\n';
  Igor::Info("N         = {}", N);
  Igor::Info("L1(T)     = {:.12e}", L1_T);
  Igor::Info("L1(r)     = {:.12e}", L1_r);
  Igor::Info("L1(r_dot) = {:.12e}", L1_r_dot);
  std::cout << '\n';
  Igor::Info("abserr(r)     = {:.12e}", r_abserr);
  Igor::Info("abserr(r_dot) = {:.12e}", r_dot_abserr);
  Igor::Info("abserr(beta)  = {:.12e}", beta_abserr);
  Igor::Info("relerr(r)     = {:.12e}", r_relerr);
  Igor::Info("relerr(r_dot) = {:.12e}", r_dot_relerr);
  Igor::Info("relerr(beta)  = {:.12e}", beta_relerr);
  std::cout << '\n';

  if (L1_T > 1.1 * Expected::L1_T(N) || std::isnan(L1_T)) {
    Igor::Error("T error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1_T(N),
                L1_T);
    any_failed = true;
  }

  if (L1_r > 1.1 * Expected::L1_r(N) || std::isnan(L1_T)) {
    Igor::Error("r error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1_r(N),
                L1_r);
    any_failed = true;
  }

  if (L1_r_dot > 1.1 * Expected::L1_r_dot(N) || std::isnan(L1_T)) {
    Igor::Error("r_dot error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1_r_dot(N),
                L1_r_dot);
    any_failed = true;
  }

  return any_failed ? 1 : 0;
}
