#include <numbers>

#ifdef USE_ALE
#include "ALE.hpp"
#endif  // USE_ALE
#include "Advection-Diffusion.hpp"
#include "BoundaryConditions.hpp"
#include "Common.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"
#include "Monitor.hpp"

// =================================================================================================
using Float               = double;
constexpr Float pi        = std::numbers::pi_v<Float>;

constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;
constexpr Float r_min     = 1.0;
constexpr Float r_max     = 2.0;
constexpr Index N         = 64;

constexpr Float D         = 1e-3;
constexpr Float CFL       = 0.5;
constexpr Float tend      = 1.0;
constexpr Float dt_write  = tend / 100.0;

#ifdef USE_ALE
constexpr Vec2<Float> w{.x = 0, .y = 1.0};
#else
constexpr None w;
#endif  // USE_ALE

// =================================================================================================
constexpr auto coords2str(Coordinates coords) -> const char* {
  switch (coords) {
    case Coordinates::CARTESIAN:           return "Cartesian";
    case Coordinates::POLAR:               return "Polar";
    case Coordinates::SYMMETRIC_SPHERICAL: return "Symmetric-Spherical";
  }
}

// =================================================================================================
auto main(int argc, char** argv) -> int {
  using namespace std::string_literals;

  const auto usage_str = Igor::detail::format(
      "Usage: {} <Coordinates>\n"
      "  Coordinates: Which coordinate system to use, choices are Cartesian, Polar, and Spherical",
      argv[0]);
  if (argc < 2) {
    Igor::Error("{}", usage_str);
    Igor::Error("Did not provide coordinates.");
    return 1;
  }

  const std::string_view coords_str = argv[1];
  Coordinates coords;
  if (coords_str == "Cartesian" || coords_str == "cartesian") {
    coords = Coordinates::CARTESIAN;
  } else if (coords_str == "Polar" || coords_str == "polar") {
    coords = Coordinates::POLAR;
  } else if (coords_str == "Symmetric-Spherical" || coords_str == "symmetric-spherical") {
    coords = Coordinates::SYMMETRIC_SPHERICAL;
  } else {
    Igor::Error("{}", usage_str);
    Igor::Error("Invalid option for coordinates `{}`", coords_str);
    return 1;
  }

  Igor::Info("{}", coords2str(coords));
#ifdef USE_ALE
  const auto case_name = "ALE-Scalar-Source-"s + coords2str(coords);
#else
  const auto case_name = "Scalar-Source-"s + coords2str(coords);
#endif  // USE_ALE
  const auto output_dir = "./test/output/"s + case_name + "/"s;
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(theta_min, theta_max, N, r_min, r_max, N, 3, coords);
#ifdef USE_ALE
  auto J_old = grid.alloc_scalar();
  auto J     = grid.alloc_scalar();
  calc_H(grid, J);
#else
  auto J_old = None{};
  auto J     = None{};
#endif  // USE_ALE

  auto src   = grid.alloc_scalar();

  auto s_old = grid.alloc_scalar();
  auto s     = grid.alloc_scalar();
  auto Fs    = grid.alloc_face_vector();

  auto u     = grid.alloc_face_vector();
#ifdef USE_ALE
  fill(u.x, w.x);
  fill(u.y, w.y);
#endif  // USE_ALE

  Float t  = 0.0;
  Float dt = 0.0;

  grid.foreach_i(FOREACH_FUNC {
    const auto r = grid.rm(j);
    if (1.25 <= r && r <= 1.75) { src(i, j) = 1.0; }
  });
  const auto total_src = stats(grid, src).sum;

  HDFWriter writer(output_dir, grid);
  writer.add_field("s", s);
#ifdef USE_ALE
  writer.add_field("J", J);
#endif  // USE_ALE
  writer.add_field("src", src);
  if (!writer.write(t)) { return 1; }

#ifdef USE_ALE
  auto stats_J = stats(grid, J);
#endif  // USE_ALE
  auto stats_s   = stats(grid, s);
  auto stats_src = stats(grid, src);

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
#ifdef USE_ALE
  monitor.add_variable(&stats_J.min, "min(J)");
  monitor.add_variable(&stats_J.max, "max(J)");
#endif  // USE_ALE
  monitor.add_variable(&stats_s.min, "min(s)");
  monitor.add_variable(&stats_s.max, "max(s)");
  monitor.add_variable(&stats_s.sum, "sum(s)");
  monitor.add_variable(&stats_src.min, "min(src)");
  monitor.add_variable(&stats_src.max, "max(src)");
  monitor.add_variable(&stats_src.sum, "sum(src)");
  monitor.write();

  BConds<Float> bconds{
      .left   = Neumann(),
      .right  = Neumann(),
      .bottom = Neumann(),
      .top    = Neumann(),
  };

  [[maybe_unused]] Float sum_dt_cube = 0.0;
  while (t < tend) {
    dt = std::min({advection_adjust_dt(grid, D, CFL), dt_write, tend - t});

#ifdef USE_ALE
    copy(J, J_old);
#endif  // USE_ALE
    copy(s, s_old);

    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

#ifdef USE_ALE
      update_J(grid, local_dt, w, J_old, J);
#endif  // USE_ALE

      calc_advection_flux(grid, u, s, w, D, Fs);
      update_s(grid, local_dt, J_old, J, Fs, src, s_old, s);
      apply_bconds(grid, bconds, s, t);

#ifdef USE_ALE
      grid.move_grid_by_velocity(w, 0.5 * dt);
#endif  // USE_ALE
    }

#ifdef USE_ALE
    stats_J = stats(grid, J);
#endif  // USE_ALE
    stats_s      = stats(grid, s);
    stats_src    = stats(grid, src);

    sum_dt_cube += dt * dt * dt;
    t           += dt;

    monitor.write();
    if (should_save(t, dt, dt_write, tend)) {
      writer.update_grid(grid);
      if (!writer.write(t)) { return 1; }
    }
  }

#ifdef USE_ALE
  Float tol;
  Float s_sum_exp;
  switch (coords) {
    case Coordinates::CARTESIAN:
      s_sum_exp = total_src * tend;
      tol       = 1e-12;
      break;
    case Coordinates::POLAR:
      {
        const auto A = grid.transform_reduce_i(
            0.0, FOREACH_FUNC { return src(i, j) * grid.dr() * grid.dtheta(); }, std::plus<>{});
        s_sum_exp = total_src * tend + 0.5 * w.r() * A * Igor::sqr(tend);
        tol       = 1e-12;
      }
      break;
    case Coordinates::SYMMETRIC_SPHERICAL:
      {
        Grid<Float> grid2(theta_min, theta_max, N, r_min, r_max, N, 3, coords);
        const auto Q_prime =
            2.0 * w.r() *
            grid2.transform_reduce_i(
                0.0,
                FOREACH_FUNC {
                  const auto theta = grid2.thetam(i);
                  const auto r     = grid2.rm(j);
                  return src(i, j) * std::sin(theta) * r * grid2.dtheta() * grid2.dr();
                },
                std::plus<>{});
        const auto Q_prime_prime =
            2.0 * Igor::sqr(w.r()) *
            grid2.transform_reduce_i(
                0.0,
                FOREACH_FUNC {
                  const auto theta = grid2.thetam(i);
                  return src(i, j) * std::sin(theta) * grid2.dtheta() * grid2.dr();
                },
                std::plus<>{});
        s_sum_exp = total_src * tend +             //
                    0.5 * Q_prime * tend * tend +  //
                    1.0 / 6.0 * Q_prime_prime * tend * tend * tend;
        tol       = 1.1 * Q_prime_prime / 24.0 * sum_dt_cube;
        break;
      }
  }
#else
  constexpr Float tol  = 1e-12;
  const auto s_sum_exp = total_src * tend;
#endif  // USE_ALE
  Igor::Info("sum(s)          = {:.12e}", stats_s.sum);
  Igor::Info("expected sum(s) = {:.12e}", s_sum_exp);
  Igor::Info("tol             = {:.12e}", tol);

  if (std::abs(stats_s.sum - s_sum_exp) > tol || std::isnan(stats_s.sum)) {
    Igor::Error("Did not get exact s, expected {:.12e} but got {:.12e}: abs. error = {:.12e}",
                s_sum_exp,
                stats_s.sum,
                std::abs(stats_s.sum - s_sum_exp));
    return 1;
  }

  Igor::Info("Ok.");
}
