#include <numbers>

#include "ALE.hpp"
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

constexpr Float tend      = 1.0;
constexpr Float dt_write  = tend / 100.0;

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_H_rel(const Grid<Float, LAYOUT>& grid,
                          const Scalar<Float, LAYOUT> H0,
                          Scalar<Float, LAYOUT> H_rel) {
  calc_H(grid, H_rel);
  grid.foreach_i(FOREACH_FUNC { H_rel(i, j) /= H0(i, j); });
}

template <typename Float, Layout LAYOUT>
constexpr void calc_H_rel(const Grid<Float, LAYOUT>& grid,
                          const Scalar<Float, LAYOUT> H0,
                          Scalar<Float, LAYOUT> H_rel) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN: return calc_H_rel<Metric::Cartesian>(grid, H0, H_rel);
    case Coordinates::POLAR:     return calc_H_rel<Metric::Polar>(grid, H0, H_rel);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return calc_H_rel<Metric::SymmetricSpherical>(grid, H0, H_rel);
  }
}

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
  const auto output_dir = "./test/output/GCL-"s + coords2str(coords) + "/"s;
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(theta_min, theta_max, N, r_min, r_max, N, 1, coords);
  auto J_old = grid.alloc_scalar();
  auto J     = grid.alloc_scalar();

  auto H     = grid.alloc_scalar();
  auto J_err = grid.alloc_scalar();

  Float t    = 0.0;
  Float dt   = 1e-2;

  calc_H(grid, J);

  calc_H(grid, H);

  const Vec2<Float> w{.x = 0.0, .y = 1.0};

  HDFWriter writer(output_dir, grid);
  writer.add_field("J", J);
  writer.add_field("H", H);
  writer.add_field("J_err", J_err);
  if (!writer.write(t)) { return 1; }

  auto stats_J      = stats(grid, J);
  auto stats_H      = stats(grid, H);
  auto stats_J_err  = stats(grid, J_err);
  auto absmax_J_err = std::max(std::abs(stats_J_err.min), std::abs(stats_J_err.max));

  Monitor<Float> monitor(output_dir + "/monitor.log");
  monitor.add_variable(&t, "t");
  monitor.add_variable(&dt, "dt");
  monitor.add_variable(&stats_J.min, "min(J)");
  monitor.add_variable(&stats_J.max, "max(J)");
  monitor.add_variable(&stats_J.sum, "sum(J)");
  monitor.add_variable(&stats_H.min, "min(H)");
  monitor.add_variable(&stats_H.max, "max(H)");
  monitor.add_variable(&stats_H.sum, "sum(H)");
  monitor.add_variable(&absmax_J_err, "absmax(J_err)");
  monitor.write();

  while (t < tend) {
    dt = std::min(dt, tend - t);

    copy(J, J_old);

    for (Index sub_iter = 0; sub_iter < 2; ++sub_iter) {
      const auto local_dt = sub_iter == 0 ? dt / 2.0 : dt;

      update_J(grid, local_dt, w, J_old, J);
      grid.move_grid_by_velocity(w, 0.5 * dt);
    }
    calc_H(grid, H);
    grid.foreach_i(FOREACH_FUNC { J_err(i, j) = J(i, j) - H(i, j); });

    stats_J       = stats(grid, J);
    stats_H       = stats(grid, H);
    stats_J_err   = stats(grid, J_err);
    absmax_J_err  = std::max(std::abs(stats_J_err.min), std::abs(stats_J_err.max));

    t            += dt;

    monitor.write();
    if (should_save(t, dt, dt_write, tend)) {
      writer.update_grid(grid);
      if (!writer.write(t)) { return 1; }
    }
  }

  const auto L1 = grid.transform_reduce_i(
      0.0, FOREACH_FUNC { return std::abs(J(i, j) - H(i, j)) * grid.dv(i, j); }, std::plus<>{});
  Igor::Info("L1(J) = {:.12e}", L1);

  constexpr Float tol = 1e-12;
  if (L1 > tol) {
    Igor::Error("Did not get exact `J`, L1-error is {:.12e}, expected <{}", L1, tol);
    return 1;
  }

  Igor::Info("Ok.");
}
