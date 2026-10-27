#include <charconv>
#include <numbers>

#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#define PALE_ASLAM_WENO
#include "ALE.hpp"
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
#ifndef DIMENSION
#define DIMENSION 2  // NOLINT
#endif               // DIMENSION

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
// constexpr Float cpg = 2.2177e3;  // Specific heat capacity    [J/(kg K)]
// constexpr Float kappag = 0.011415;               // Thermal conductivity      [W/(m K)]
// constexpr Float alphag = kappag / (rhog * cpg);  // Thermal diffusivity       [m^2/s]

// Temperatures
constexpr Float Tsat = 111.0;  // Saturation temperature    [K]
// constexpr Float Tinf     = 111.26;    // Bulk temperature          [K]
constexpr Float hev                 = 5.1083e5;  // Enthalpy of vaporization  [J/kg]

constexpr Float eps                 = (rhol - rhog) / rhol;  // Relative difference of density [-]

constexpr Float dTdn                = 1e3;  // [K/m]

constexpr Float CFL                 = 0.5;
constexpr Float tend                = 1.0;
constexpr Float dt_write            = tend / 100.0;

constexpr Index EXTRAPOLATION_ORDER = 1;

HalfPlane<Float> wall{.x = -1.5 * r_min, .y = 0.0, .nx = -1.0, .ny = 0.0};
// = Setup =========================================================================================

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
    const auto A_bot = Metric::H(grid.xm(i), grid.y(0)) / Metric::h2(grid.xm(i), grid.y(0));
    const auto A_top = Metric::H(grid.xm(i), grid.y(jtop)) / Metric::h2(grid.xm(i), grid.y(jtop));
    /* if (!wall.contains(polar2cartesian(grid.xm(i), grid.y(0)))) */ { Q_in += u.y(i, 0) * A_bot; }
    /* if (!wall.contains(polar2cartesian(grid.xm(i), grid.y(jtop)))) */ {
      Q_out += u.y(i, jtop) * A_top;
    }
    if (!wall.contains(polar2cartesian(grid.xm(i), grid.y(jtop)))) { A_out += A_top; }
  }
  const Float corr = (Q_in - Q_out) / A_out;
  for (Index i = 0; i < u.y.nx(); ++i) {
    if (!wall.contains(polar2cartesian(grid.xm(i), grid.y(jtop)))) { u.y(i, jtop) += corr; }
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
      [=](Index i, Index j) -> Float {
        if (wall.contains(polar2cartesian(grid.thetam(i), grid.rm(j)))) { return 0.0; }

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
        if (wall.contains(polar2cartesian(grid.thetam(i), grid.rm(j)))) { return 0.0; }

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
template <typename Float, Layout LAYOUT>
constexpr void
set_solid_cells(const Grid<Float, LAYOUT>& grid,
                const FaceVector<Float, LAYOUT> ib_corr,
                Float rho,
                Float mu,
                const Vec2<Float>& u_wall_cartesian,  // This is **always** in cartesian coordinates
                FaceVector<Float, LAYOUT> u) {
  PALE_PROFILE_FUNCTION;

  const auto nu = mu / rho;
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:
      grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
        const auto lambda = nu * ib_corr.x(i, j);
        if (lambda > detail::LAMBDA_MAX) { u.x(i, j) = u_wall_cartesian.x; }
      });
      grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
        const auto lambda = nu * ib_corr.y(i, j);
        if (lambda > detail::LAMBDA_MAX) { u.y(i, j) = u_wall_cartesian.y; }
      });
      break;
    case Coordinates::SYMMETRIC_SPHERICAL:
    case Coordinates::POLAR:
      grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
        const auto lambda = nu * ib_corr.x(i, j);
        const auto theta  = grid.theta(i);
        const auto uth =
            -std::sin(theta) * u_wall_cartesian.x + std::cos(theta) * u_wall_cartesian.y;
        if (lambda > detail::LAMBDA_MAX) { u.x(i, j) = uth; }
      });
      grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
        const auto lambda = nu * ib_corr.y(i, j);
        const auto theta  = grid.thetam(i);
        const auto ur = std::cos(theta) * u_wall_cartesian.x + std::sin(theta) * u_wall_cartesian.y;
        if (lambda > detail::LAMBDA_MAX) { u.y(i, j) = ur; }
      });
      break;
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

  const auto output_dir = get_output_directory() + "/" + std::to_string(N);
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

  auto ib_corr_face      = grid.alloc_face_vector();
  auto ib_levelset       = grid.alloc_scalar();
  auto ib_free           = FOREACH_FUNC->Float { return ib_levelset(i, j) <= 0.0; };

  Float m_dot_total      = 0.0;
  Float r_dot            = 0.0;

  Float dt               = 0.0;
  Float t                = 0.0;

  const Float band_width = 3.0;

  calc_ib_correction_shape(grid, wall, ib_corr_face);
  calc_ib_levelset_shape(grid, wall, ib_levelset);

  Vec2<Float> w{};
  Vec2<Float> u_wall_cartesian{.x = 0.0, .y = 0.0};
  fill(u, 0.0);
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
  apply_velocity_bconds(grid, uth_bconds, ur_bconds, u);

  const BConds<Float> T_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Dirichlet<Float>{.val = Tsat},
      .top    = Neumann(),
  };
  const BConds<Float> dp_bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };
  MultigridSolver solver(grid, dp_bconds);
  // Index mg_cycles = 0;
  // Float mg_res    = 0.0;

  fill(T, Tsat);
  [[maybe_unused]] Index aslam_iter =
      neumann_extrapolate<EXTRAPOLATION_ORDER>(grid, T, ib_levelset, dTdn, band_width);

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", ui);
  writer.add_field("p", p);
  writer.add_field("T", T);
  writer.add_field("div", div);
  writer.add_field("psi", ib_levelset);
  if (!writer.write(t)) { return 1; }

  Stats p_stats   = stats(grid, p);
  Stats u_stats   = stats(grid, u.x);
  Stats v_stats   = stats(grid, u.y);
  Stats T_stats   = stats(grid, T, ib_free);
  Stats div_stats = stats(grid, div);
  Float div_max   = std::max(std::abs(div_stats.min), std::abs(div_stats.max));

  Float r         = grid.r_min();

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&p_stats.max, "max(p)");
  monitor.add_variable(&u_stats.max, "max(u_theta)");
  monitor.add_variable(&v_stats.max, "max(u_r)");
  monitor.add_variable(&m_dot_total, "m_dot_total");
  monitor.add_variable(&r_dot, "r_dot");
  monitor.add_variable(&T_stats.min, "min(T)");
  monitor.add_variable(&T_stats.max, "max(T)");
  monitor.add_variable(&T_stats.sum, "sum(T)");
  monitor.add_variable(&r, "r");
  monitor.add_variable(&wall.x, "wall(x)");
  // monitor.add_variable(&u_wall_cartesian.x, "wall(ux)");
  monitor.add_variable(&div_max, "absmax(div)");
  // monitor.add_variable(&mg_res, "res(MG)");
  // monitor.add_variable(&mg_cycles, "cycles(MG)");
  // monitor.add_variable(&aslam_iter, "iter(Aslam)");
  monitor.write();

  IGOR_TIME_SCOPE("Solver")
  while (t < tend) {
    dt = std::min({
        adjust_dt(grid, u, rhol, mul, CFL),
        advection_adjust_dt(grid, alphal, CFL),
        ale_adjust_dt(grid, w, CFL),
        dt_write,
        tend - t,
    });

    copy(u, u_old);
    copy(T, T_old);
    copy(J, J_old);

    // mg_cycles  = 0;
    aslam_iter = 0;
    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      // 1) Mass exchange -> grid velocity
      m_dot_total        = calc_m_dot_total(grid, T);
      r_dot              = calc_r_dot(grid.y_min(), m_dot_total);
      w.r()              = r_dot;
      ur_bconds.bottom   = Dirichlet<Float>{.val = eps * w.r()};
      u_wall_cartesian.x = -r_dot;

      // 2) Update J
      update_J(grid, local_dt, w, J_old, J);

      // 3) Predictor
      calc_mom_flux(grid, u, p, rhol, mul, w, FUX, FUY, FVX, FVY, FWZ);
      update_u(grid, local_dt, J_old, J, FUX, FUY, FVX, FVY, FWZ, u_old, u);
      correct_velocity_ib_implicit_euler(
          grid, ib_corr_face, rhol, mul, local_dt, u_wall_cartesian, u);
      apply_velocity_bconds(grid, uth_bconds, ur_bconds, u);
      correct_outflow(grid, u);

      // 4) Pressure correction
      // 4.1) Move grid for pressure correction
      grid.move_grid_by_velocity(w, 0.5 * dt);
      solver.move_grid_by_velocity(w, 0.5 * dt);
      // 4.2) Actual correction
      calc_div(grid, u, div);
      grid.foreach_i(FOREACH_FUNC { div(i, j) *= rhol / local_dt; });
      if (!solver.solve(dp, div, 1e-6 / local_dt)) {
        Igor::Warn("t={:.8f}: Multigrid solver did not converge after {} cycles: res = {:.8e}",
                   t,
                   solver.num_cycles(),
                   solver.res());
      }
      // mg_cycles += solver.num_cycles();
      // mg_res     = solver.res();
      apply_bconds(grid, dp_bconds, dp, t);

      // 5) Project
      // 5.1) Actual projection
      correct_velocity(grid, dp, rhol, local_dt, u, p);
      // set_solid_cells(grid, ib_corr_face, rhol, mul, u_wall_cartesian, u);

      apply_velocity_bconds_only_periodic(grid, uth_bconds, ur_bconds, u);
      // 5.2) Move grid back for scalar transport
      grid.move_grid_by_velocity(w, -0.5 * dt);

      // 6) Update temperature
      calc_advection_flux(grid, u, T, w, alphal, FT);
      update_s(grid, local_dt, J_old, J, FT, T_old, T);
      apply_bconds(grid, T_bconds, T, t);
      aslam_iter = neumann_extrapolate<EXTRAPOLATION_ORDER>(grid, T, ib_levelset, dTdn, band_width);

      // 7) Update the physical position of the grid
      grid.move_grid_by_velocity(w, 0.5 * dt);
      // solver.move_grid_by_velocity(w, 0.5 * dt); // Already moved in step 4.1)

      // 8) Update correction terms
      wall.x += 0.5 * dt * u_wall_cartesian.x;
      fill(ib_corr_face, 0.0);
      calc_ib_correction_shape(grid, wall, ib_corr_face);
      calc_ib_levelset_shape(grid, wall, ib_levelset);
    }
    interpolate(grid, u, ui);
    calc_div(grid, u, div);

    p_stats    = stats(grid, p);
    u_stats    = stats(grid, u.x);
    v_stats    = stats(grid, u.y);
    T_stats    = stats(grid, T, ib_free);
    div_stats  = stats(grid, div);
    div_max    = std::max(std::abs(div_stats.min), std::abs(div_stats.max));
    r          = grid.r_min();

    t         += dt;
    if (should_save(t, dt, dt_write, tend)) {
      writer.update_grid(grid);
      if (!writer.write(t)) { return 1; }
    }
    monitor.write();
  }

  Igor::Info("Ok.");
}
