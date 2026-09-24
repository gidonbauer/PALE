#include <charconv>
#include <numbers>

#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Advection-Diffusion.hpp"
#include "BoundaryConditions.hpp"
#include "Common.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"
#include "Mac.hpp"
#include "Monitor.hpp"
#include "MultigridPoisson.hpp"

#include "Scriven-Analytical.hpp"

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
template <typename Float, Layout LAYOUT>
constexpr auto ale_adjust_dt(const Grid<Float, LAYOUT>& grid,
                             const Vec2<Float>& w_,
                             Float CFL_) noexcept -> Float {
  // Correction for polar coordinates
  const auto hx = grid.coords() == Coordinates::POLAR ? grid.ym(0) * grid.dx() : grid.dx();
  const auto hy = grid.dy();

  // Advection: dt * (|u|/hx + |v|/hy) <= CFL
  const auto adv          = std::abs(w_.x) / hx + std::abs(w_.y) / hy;
  constexpr auto no_limit = std::numeric_limits<Float>::max();
  return adv > 0.0 ? CFL_ / adv : no_limit;
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
void correct_outflow(const Grid<Float, LAYOUT>& grid, FaceVector<Float, LAYOUT> u) {
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
template <typename Float, Layout LAYOUT>
constexpr auto calc_m_dot_total(const Grid<Float, LAYOUT>& grid, Scalar<Float, LAYOUT> T) -> Float {
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
}

template <typename Float>
constexpr auto calc_r_dot(Float r, Float m_dot_total) -> Float {
  // 2D:
  return m_dot_total / (2.0 * pi * rhog * r);
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

  const auto output_dir = get_output_directory();
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(theta_min, theta_max, N / 2, r_min, r_max, N, 3, Coordinates::POLAR);
  auto u_old = grid.alloc_face_vector();
  auto u     = grid.alloc_face_vector();
  auto ui    = grid.alloc_vector();

  auto FUX   = grid.alloc_scalar();
  auto FUY   = grid.alloc_vertex_scalar();
  auto FVX   = grid.alloc_vertex_scalar();
  auto FVY   = grid.alloc_scalar();

  auto div   = grid.alloc_scalar();
  auto p     = grid.alloc_scalar();
  auto dp    = grid.alloc_scalar();

  auto T_old = grid.alloc_scalar();
  auto T     = grid.alloc_scalar();
  auto FT    = grid.alloc_face_vector();

  Scriven::Params params{
      .Ja        = Ja,
      .eps       = eps,
      .alpha     = alphal,
      .Tsat      = Tsat,
      .Tinf      = Tinf,
      .beta      = 0.0,
      .dimension = 2,
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

  const BConds<Float> s_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = Tsat},
      .top    = Dirichlet<Float>{.val = Tinf},
  };
  grid.foreach_i(FOREACH_FUNC mutable { T(i, j) = Scriven::T(grid.rm(j), t, params); });
  apply_bconds(grid, s_bconds, T, t);

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

  // Float mg_res      = 0.0;
  // Index mg_cycles   = 0;

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  // monitor.add_variable(&p_stats.max, "max(p)");
  // monitor.add_variable(&u_stats.max, "max(u_theta)");
  monitor.add_variable(&v_stats.max, "max(u_r)");
  monitor.add_variable(&T_stats.min, "min(T)");
  monitor.add_variable(&T_stats.max, "max(T)");
  monitor.add_variable(&m_dot_total, "m_dot_total");
  monitor.add_variable(&r_dot, "r_dot");
  monitor.add_variable(&r, "r");
  monitor.add_variable(&beta_eff, "beta_eff");
  monitor.add_variable(&r_abserr, "abserr(r)");
  monitor.add_variable(&r_dot_abserr, "abserr(r_dot)");
  monitor.add_variable(&beta_abserr, "abserr(beta)");
  monitor.add_variable(&r_relerr, "relerr(r)");
  monitor.add_variable(&r_dot_relerr, "relerr(r_dot)");
  monitor.add_variable(&beta_relerr, "relerr(beta)");
  monitor.add_variable(&div_max, "absmax(div)");
  // monitor.add_variable(&mg_res, "res(MG)");
  // monitor.add_variable(&mg_cycles, "cycles(MG)");
  monitor.write();
  // - Output ------------------------------------------------------------------

  IGOR_TIME_SCOPE("Solver")
  while (t < tend) {
    dt = std::min({
        adjust_dt(grid, u, rhol, mul, CFL),
        advection_adjust_dt(grid, alphal, CFL),
        ale_adjust_dt(grid, w, CFL),
        dt_write,
        std::max(tend - t, 1e-6),
    });

    copy(u, u_old);
    copy(T, T_old);

    // mg_cycles = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? 0.5 * dt : dt;

      // 1) Mass exchange -> grid velocity
      m_dot_total      = calc_m_dot_total(grid, T);
      r_dot            = calc_r_dot(grid.y_min(), m_dot_total);
      w.r()            = r_dot;
      ur_bconds.bottom = Dirichlet<Float>{.val = eps * w.r()};

      // 2) Prediction
      ALEPolar::calc_mom_flux(grid, u, p, rhol, mul, w, FUX, FUY, FVX, FVY);
      ALEPolar::update_u(grid, local_dt, w, FUX, FUY, FVX, FVY, u_old, u);
      apply_velocity_bconds(grid, uth_bconds, ur_bconds, u);

      // 3) Update the physical position of the grid
      grid.move_grid_by_velocity(w, 0.5 * dt);
      solver.move_grid_by_velocity(w, 0.5 * dt);
      correct_outflow(grid, u);

      // 4) Pressure calculation
      ALEPolar::calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rhol / local_dt; });
      if (!solver.solve(dp, div, 1e-3 / local_dt)) {
        Igor::Warn("t={:.8f}: Multigrid solver did not converge after {} cycles: res = {:.8e}",
                   t,
                   solver.num_cycles(),
                   solver.res());
      }
      // mg_res     = solver.res();
      // mg_cycles += solver.num_cycles();
      apply_bconds(grid, dp_bconds, dp, t);

      // 5) Projection
      ALEPolar::correct_velocity(grid, dp, rhol, local_dt, u, p);
      apply_velocity_bconds_only_periodic(grid, uth_bconds, ur_bconds, u);

      // 6) Update temperature
      ALEPolar::calc_advection_flux(grid, u, T, w, alphal, FT);
      ALEPolar::update_s(grid, local_dt, w, FT, T_old, T);
      apply_bconds(grid, s_bconds, T, t);
    }
    ALEPolar::calc_div(grid, u, div);
    interpolate(grid, u, ui);

    p_stats       = stats(grid, p);
    u_stats       = stats(grid, u.x);
    v_stats       = stats(grid, u.y);
    div_stats     = stats(grid, div);
    T_stats       = stats(grid, T);
    div_max       = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

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

    t            += dt;
    monitor.write();
    if (should_save(t, dt, dt_write, tend)) {
      writer.update_grid(grid);
      if (!writer.write(t)) { return 1; }
    }
  }

  Igor::Info("Ok.");
}
