#include <charconv>
#include <cmath>
#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Common.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"
#include "MultigridPoisson.hpp"

#include "Test-Common.hpp"

// = Setup =========================================================================================
using Float               = double;
constexpr auto pi         = std::numbers::pi_v<Float>;

constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi;
constexpr Float r_min     = 1.0;
constexpr Float r_max     = 2.0;

constexpr Float tol       = 1e-8;
// = Setup =========================================================================================

// =================================================================================================
// Manufactured solution in rotationally symmetric spherical coordinates:
//   F(theta, r) = (1 + cos(theta)) * f(r),  f(r) = cos(pi (r - 1))
// Satisfies homogeneous Neumann conditions at r = r_min, r = r_max and at the axis theta = 0, pi.
//   L F = 1/r^2 d/dr(r^2 dF/dr) + 1/(r^2 sin(theta)) d/dtheta(sin(theta) dF/dtheta)
//       = (f" + 2/r f') + cos(theta) * (f" + 2/r f' - 2/r^2 f)
// The radial part has a non-zero mean for a wrong cell volume, e.g. dV = r dr dtheta; the solver
// then cannot make the right-hand side compatible with the Neumann conditions and diverges.
constexpr auto f(Float r) -> Float { return std::cos(pi * (r - r_min)); }
constexpr auto df(Float r) -> Float { return -pi * std::sin(pi * (r - r_min)); }
constexpr auto ddf(Float r) -> Float { return -Igor::sqr(pi) * std::cos(pi * (r - r_min)); }

constexpr auto F(Float theta, Float r) -> Float { return (1.0 + std::cos(theta)) * f(r); }
constexpr auto LF(Float theta, Float r) -> Float {
  return (ddf(r) + 2.0 / r * df(r)) +
         std::cos(theta) * (ddf(r) + 2.0 / r * df(r) - 2.0 / Igor::sqr(r) * f(r));
}

// =================================================================================================
// Volume weighted L1 error, normalized by the volume of the domain
template <typename Float, Layout LAYOUT>
constexpr auto L1error(const Grid<Float, LAYOUT>& grid,
                       const Scalar<Float, LAYOUT> f_true,
                       const Scalar<Float, LAYOUT> f_pred) -> Float {
  const Float err = grid.transform_reduce_i(
      0.0,
      FOREACH_FUNC { return std::abs(f_true(i, j) - f_pred(i, j)) * grid.dv(i, j); },
      std::plus<>{});
  const Float vol =
      grid.transform_reduce_i(0.0, FOREACH_FUNC { return grid.dv(i, j); }, std::plus<>{});
  return err / vol;
}

// =================================================================================================
namespace Expected {
constexpr std::array ns  = {16, 32, 64, 128, 256};
constexpr std::array L1s = {
    4.71143208e-03,
    1.16771553e-03,
    2.91279644e-04,
    7.27799857e-05,
    1.81924915e-05,
};
static_assert(ns.size() == L1s.size());
constexpr auto L1(Index n) { return interp_n2(ns, L1s, n); }
}  // namespace Expected

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
  Grid<Float> grid(theta_min, theta_max, N, r_min, r_max, N, 1, Coordinates::SYMMETRIC_SPHERICAL);

  auto rhs   = grid.alloc_scalar();
  auto f_exp = grid.alloc_scalar();
  auto f_mg  = grid.alloc_scalar();

  grid.foreach_i(FOREACH_FUNC { rhs(i, j) = LF(grid.thetam(i), grid.rm(j)); });
  grid.foreach_i(FOREACH_FUNC { f_exp(i, j) = F(grid.thetam(i), grid.rm(j)); });
  // The solver returns the solution with zero mean
  const auto f_exp_stats = stats(grid, f_exp);
  const Float f_exp_mean = f_exp_stats.sum / f_exp_stats.volume;
  grid.foreach_i(FOREACH_FUNC { f_exp(i, j) -= f_exp_mean; });

  bool any_failed = false;
  Float L1_mg     = -1.0;
  {
    Igor::ScopeTimer timer("Multigrid-Spherical-" + std::to_string(N));
    MultigridSolver solver(grid);
    const bool converged = solver.solve(f_mg, rhs, tol, 1'000);
    L1_mg                = L1error(grid, f_exp, f_mg);
    Igor::Info("cycles = {}, res = {:.4e}", solver.num_cycles(), solver.res());
    if (!converged) {
      Igor::Error("Multigrid solver did not converge.");
      any_failed = true;
    }
  }

  Igor::Info("L1({})     = {:.8e}", N, L1_mg);
  Igor::Info("L1_exp({}) = {:.8e}", N, Expected::L1(N));
  if (L1_mg > 1.1 * Expected::L1(N) || std::isnan(L1_mg)) {
    Igor::Error("Error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1(N),
                L1_mg);
    any_failed = true;
  }

  const auto output_dir = "./test/output/Multigrid-Spherical-" + std::to_string(N);
  if (!init_output_directory(output_dir)) { return 1; }
  HDFWriter writer(output_dir, grid);
  writer.add_field("f_exp", f_exp);
  writer.add_field("f_mg", f_mg);
  if (!writer.write(0.0)) { return 1; }

  return any_failed ? 1 : 0;
}
