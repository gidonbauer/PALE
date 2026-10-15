#include <charconv>
#include <numbers>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "AslamExtrapolation.hpp"
#include "Geometry.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"

// =================================================================================================
using Float        = double;
constexpr Float pi = std::numbers::pi_v<Float>;

constexpr Float L  = 2.0 * pi;
constexpr Circle<Float> c{.x = 0.0, .y = 0.0, .r = 2.0};

// =================================================================================================
constexpr auto u_func(Float x, Float y) -> Float {
  // return std::sqrt(Igor::sqr(x - c.x) + Igor::sqr(y - c.y));
  return Igor::sqr(x - c.x) + Igor::sqr(y - c.y);
  // constexpr Float k = 3.0;
  // const auto r      = std::hypot(x - c.x, y - c.y);
  // const auto theta  = std::atan2(y - c.y, x - c.x);
  // return (1.0 + r) * std::cos(k * theta);
}

constexpr void
get_normal_and_intersection(Float x, Float y, Vec2<Float>& n, Vec2<Float>& i, Float& d) {
  n = Vec2<Float>{
      .x = x - c.x,
      .y = y - c.y,
  };

  d  = norm(n);
  n /= d;

  i  = {
      .x = c.r * n.x + c.x,
      .y = c.r * n.y + c.y,
  };
}

constexpr auto u_expected(double x, double y, double dudn) -> Float {
  if (Igor::sqr(x - c.x) + Igor::sqr(y - c.y) < Igor::sqr(c.r)) { return u_func(x, y); }

  Vec2<Float> n{};
  Vec2<Float> i{};
  Float d{};
  get_normal_and_intersection(x, y, n, i, d);

  Float ui = u_func(i.x, i.y);
  return ui + dudn * (d - c.r);
}

// =================================================================================================
auto main(int argc, char** argv) -> int {
  const auto usage_str = Igor::detail::format("Usage: {} <grid size>", argv[0]);

  if (argc < 2) {
    Igor::Error("{}", usage_str);
    Igor::Error("  Did not provide grid size.");
    return 1;
  }

  Index N;
  if (std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), N).ec != std::errc{} || N <= 0) {
    Igor::Error("{}", usage_str);
    Igor::Error("  Invalid grid size `{}`", argv[1]);
    return 1;
  }

  const auto output_dir = get_output_directory();
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(-L / 2.0, L / 2.0, N, -L / 2.0, L / 2.0, N, 3);

  auto u           = grid.alloc_scalar();  // To-be-extrapolated value
  auto u_exp       = grid.alloc_scalar();
  auto psi         = grid.alloc_scalar();  // Levelset
  auto dudn_calc   = grid.alloc_scalar();

  const Float dudn = 2.0 * c.r;

  // ===============================================================================================
  grid.foreach_a(FOREACH_FUNC {
    const auto x = grid.xm(i);
    const auto y = grid.ym(j);
    psi(i, j)    = c.levelset(x, y);
    u(i, j)      = psi(i, j) > 0.0 ? 0.0 : u_func(x, y);
    u_exp(i, j)  = u_expected(x, y, dudn);
  });

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", u);
  writer.add_field("u_exp", u_exp);
  writer.add_field("psi", psi);
  writer.add_field("dudn", dudn_calc);
  writer.write(0.0);

  const double band_size = 6.0 * std::max(grid.dx(), grid.dy());
#if 1
  const double tau_end = 2.5;
#else
  const double tau_end = 4.0 * band_size;
#endif
  IGOR_TIME_SCOPE("Extrapolation") { neumann_extrapolate(grid, u, psi, dudn, tau_end); }

  grid.foreach_i(FOREACH_FUNC {
    if (psi(i, j) < 0.0) {
      dudn_calc(i, j) = std::numeric_limits<Float>::quiet_NaN();
      return;
    }

    const auto x    = grid.xm(i);
    const auto y    = grid.ym(j);
    const auto n    = normalize(Vec2<Float>{
        .x = x - c.x,
        .y = y - c.y,
    });

    const auto dudx = [=] {
      if (0 < i && i < grid.nx() - 1) {
        // Central: -1/2 0 1/2
        return (u(i + 1, j) - u(i - 1, j)) / (2.0 * grid.dx());
      } else if (i <= 0) {
        // Forwards: −3/2 4/2 −1/2
        return (-3.0 * u(i, j) + 4.0 * u(i + 1, j) - u(i + 2, j)) / (2.0 * grid.dx());
      } else {
        // Backwards: 1/2 −4/2 3/2
        return (u(i - 2, j) - 4.0 * u(i - 1, j) + 3.0 * u(i, j)) / (2.0 * grid.dx());
      }
    }();
    const auto dudy = [=] {
      if (0 < j && j < grid.ny() - 1) {
        // Central: -1/2 0 1/2
        return (u(i, j + 1) - u(i, j - 1)) / (2.0 * grid.dy());
      } else if (j <= 0) {
        // Forwards: −3/2 4/2 −1/2
        return (-3.0 * u(i, j) + 4.0 * u(i, j + 1) - u(i, j + 2)) / (2.0 * grid.dy());
      } else {
        // Backwards: 1/2 −4/2 3/2
        return (u(i, j - 2) - 4.0 * u(i, j - 1) + 3.0 * u(i, j)) / (2.0 * grid.dy());
      }
    }();
    dudn_calc(i, j) = n.x * dudx + n.y * dudy;
  });

  writer.write(1.0);

  struct Err {
    Float L1;
    Float L1_band;
    Float vol_band;
  };

  Err e_u = grid.transform_reduce_i(
      Err{},
      FOREACH_FUNC {
        const auto in_band  = 0.0 <= psi(i, j) && psi(i, j) <= band_size;
        const auto abs_diff = std::abs(u(i, j) - u_exp(i, j)) * grid.dv(i, j);
        return Err{
            .L1       = abs_diff,
            .L1_band  = in_band ? abs_diff : 0.0,
            .vol_band = in_band ? grid.dv(i, j) : 0.0,
        };
      },
      [](Err lhs, const Err& rhs) {
        lhs.L1       += rhs.L1;
        lhs.L1_band  += rhs.L1_band;
        lhs.vol_band += rhs.vol_band;
        return lhs;
      });
  e_u.L1_band /= e_u.vol_band;

  Err e_dudn   = grid.transform_reduce_i(
      Err{},
      FOREACH_FUNC {
        const auto in_band   = 0.0 <= psi(i, j) && psi(i, j) <= band_size;
        const auto in_liquid = 0.0 <= psi(i, j);
        const auto abs_diff  = in_liquid ? std::abs(dudn_calc(i, j) - dudn) * grid.dv(i, j) : 0.0;
        return Err{
            .L1       = abs_diff,
            .L1_band  = in_band ? abs_diff : 0.0,
            .vol_band = in_band ? grid.dv(i, j) : 0.0,
        };
      },
      [](Err lhs, const Err& rhs) {
        lhs.L1       += rhs.L1;
        lhs.L1_band  += rhs.L1_band;
        lhs.vol_band += rhs.vol_band;
        return lhs;
      });
  e_dudn.L1_band /= e_dudn.vol_band;

  Igor::Info("Primal:     L1({})      = {:.12e}", N, e_u.L1);
  Igor::Info("Primal:     L1_band({}) = {:.12e}", N, e_u.L1_band);
  Igor::Info("Derivative: L1({})      = {:.12e}", N, e_dudn.L1);
  Igor::Info("Derivative: L1_band({}) = {:.12e}", N, e_dudn.L1_band);
}
