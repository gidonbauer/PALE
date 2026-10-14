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

#include "Test-Common.hpp"

#ifndef EXTRAPOLATION_ORDER
#define EXTRAPOLATION_ORDER 0
#endif  // EXTRAPOLATION_ORDER

#if EXTRAPOLATION_ORDER < 0 || EXTRAPOLATION_ORDER > 2
#error "Only constant, linear, and quadratic extrapolation is possible"
#endif  // EXTRAPOLATION_ORDER

// =================================================================================================
using Float        = double;
constexpr Float pi = std::numbers::pi_v<Float>;

constexpr Float L  = 2.0 * pi;
constexpr Circle<Float> c{.x = 0.0, .y = 0.0, .r = 2.0};

// =================================================================================================
namespace Expected {

constexpr std::array ns     = {16, 32, 64, 128, 256, 512, 1024};
constexpr std::array L1s_bc = {
    1.885017002079e+00,
    4.004804173020e-01,
    1.454171212535e-01,
    2.997703603400e-02,
    7.954459223849e-03,
    1.947892606690e-03,
    4.671124426510e-04,
};
constexpr std::array L1s_bl = {
    2.484461044747e+00,
    3.642126369932e-01,
    5.232710050122e-02,
    4.966248384123e-03,
    6.639027621351e-04,
    7.753458440821e-05,
    9.145154878916e-06,
};
constexpr std::array L1s_bq = {
    7.156618989221e+00,
    8.897251342536e-01,
    5.400456590453e-02,
    2.799026085915e-03,
    1.665369183605e-04,
    1.049372163606e-05,
    6.446166250686e-07,
};

constexpr auto L1_band(Index N) {
  switch (EXTRAPOLATION_ORDER) {
    case 0:  return interp_pow(ns, L1s_bc, N, 2);
    case 1:  return interp_pow(ns, L1s_bl, N, 3);
    case 2:  return interp_pow(ns, L1s_bq, N, 4);
    default: Igor::Panic("Invalid order `{}`", EXTRAPOLATION_ORDER);
  }
}

}  // namespace Expected

// =================================================================================================
constexpr auto u_func(Float x, Float y) -> Float { return std::cos(x) * std::sin(y); }

// Directional derivatives generated with TAPENADE (https://www-sop.inria.fr/tropics/)
constexpr auto ddudn(Float x, Float y, Vec2<Float> n, Float& u, Float& dudn) -> Float {
  const auto temp0 = std::sin(y);
  const auto temp1 = std::cos(y) * n.y;
  const auto temp2 = std::cos(x);
  const auto temp3 = -(std::sin(x) * n.x);
  const auto temp4 = std::cos(y);
  const auto temp5 = std::sin(x);

  u                = temp2 * temp0;
  dudn             = temp0 * temp3 + temp2 * temp1;
  return n.y * (temp4 * temp3 - temp2 * sin(y) * n.y) -
         n.x * (temp5 * temp1 + temp0 * cos(x) * n.x);
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

constexpr auto u_expected_constant(Float x, Float y) -> Float {
  if (Igor::sqr(x - c.x) + Igor::sqr(y - c.y) < Igor::sqr(c.r)) { return u_func(x, y); }

  Vec2<Float> n{};
  Vec2<Float> i{};
  Float d{};
  get_normal_and_intersection(x, y, n, i, d);
  return u_func(i.x, i.y);
}

constexpr auto u_expected_linear(double x, double y) -> Float {
  if (Igor::sqr(x - c.x) + Igor::sqr(y - c.y) < Igor::sqr(c.r)) { return u_func(x, y); }

  Vec2<Float> n{};
  Vec2<Float> i{};
  Float d{};
  get_normal_and_intersection(x, y, n, i, d);

  Float ui;
  Float duidn;
  ddudn(i.x, i.y, n, ui, duidn);
  return ui + duidn * (d - c.r);
}

constexpr auto u_expected_quadratic(double x, double y) -> Float {
  if (Igor::sqr(x - c.x) + Igor::sqr(y - c.y) < Igor::sqr(c.r)) { return u_func(x, y); }

  Vec2<Float> n{};
  Vec2<Float> i{};
  Float d{};
  get_normal_and_intersection(x, y, n, i, d);

  Float ui;
  Float duidn;
  Float dduidn = ddudn(i.x, i.y, n, ui, duidn);
  return ui + duidn * (d - c.r) + 0.5 * dduidn * Igor::sqr(d - c.r);
}

constexpr auto order2str() -> std::string {
  switch (EXTRAPOLATION_ORDER) {
    case 0:  return "Constant";
    case 1:  return "Linear";
    case 2:  return "Quadratic";
    default: Igor::Panic("Invalid order `{}`", EXTRAPOLATION_ORDER);
  }
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

  const auto output_dir = "./test/output/Aslam-" + order2str() + '-' + std::to_string(N);
  if (!init_output_directory(output_dir)) { return 1; }

  Grid<Float> grid(-L / 2.0, L / 2.0, N, -L / 2.0, L / 2.0, N, 3);

  auto u     = grid.alloc_scalar();  // To-be-extrapolated value
  auto u_exp = grid.alloc_scalar();
  auto psi   = grid.alloc_scalar();  // Levelset

  // ===============================================================================================
  grid.foreach_a(FOREACH_FUNC {
    const auto x = grid.xm(i);
    const auto y = grid.ym(j);
    psi(i, j)    = std::sqrt(Igor::sqr(x - c.x) + Igor::sqr(y - c.y)) - c.r;
    u(i, j)      = psi(i, j) > 0.0 ? 0.0 : u_func(x, y);
    switch (EXTRAPOLATION_ORDER) {
      case 0:  u_exp(i, j) = u_expected_constant(x, y); break;
      case 1:  u_exp(i, j) = u_expected_linear(x, y); break;
      case 2:  u_exp(i, j) = u_expected_quadratic(x, y); break;
      default: Igor::Panic("Invalid order `{}`", EXTRAPOLATION_ORDER);
    }
  });

  HDFWriter writer(output_dir, grid);
  writer.add_field("u", u);
  writer.add_field("u_exp", u_exp);
  writer.add_field("psi", psi);
  writer.write(0.0);

  const double band_size = 6.0 * std::max(grid.dx(), grid.dy());
#if 0
  const double tau_end = 2.5;
#else
  const double tau_end = 4.0 * band_size;
#endif
  IGOR_TIME_SCOPE("Extrapolation") {
    switch (EXTRAPOLATION_ORDER) {
      case 0:  constant_extrapolate(grid, u, psi, tau_end); break;
      case 1:  linear_extrapolate(grid, u, psi, tau_end); break;
      case 2:  quadratic_extrapolate(grid, u, psi, tau_end); break;
      default: Igor::Panic("Invalid order `{}`", EXTRAPOLATION_ORDER);
    }
  }
  writer.write(1.0);

  const auto L1 = grid.transform_reduce_i(
      0.0, FOREACH_FUNC { return std::abs(u(i, j) - u_exp(i, j)) * grid.dv(i, j); }, std::plus<>{});

  const auto L1_band = grid.transform_reduce_i(
      0.0,
      FOREACH_FUNC {
        if (0.0 <= psi(i, j) && psi(i, j) <= band_size) {
          return std::abs(u(i, j) - u_exp(i, j)) * grid.dv(i, j);
        }
        return 0.0;
      },
      std::plus<>{});

  Igor::Info("{} extrapolation", order2str());
  Igor::Info("L1({})      = {:.12e}", N, L1);
  Igor::Info("L1_band({}) = {:.12e}", N, L1_band);

  if (L1_band > 1.1 * Expected::L1_band(N)) {
    Igor::Error("s error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1_band(N),
                L1);
    return 1;
  }
}
