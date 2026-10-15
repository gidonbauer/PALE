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
// NOLINTNEXTLINE
#define EXTRAPOLATION_ORDER 0
#endif  // EXTRAPOLATION_ORDER

#if EXTRAPOLATION_ORDER < 0 || EXTRAPOLATION_ORDER > 2
#error "Only constant, linear, and quadratic extrapolation is possible"
#endif  // EXTRAPOLATION_ORDER

// =================================================================================================
using Float        = double;
constexpr Float pi = std::numbers::pi_v<Float>;

constexpr Float L  = 2.0 * pi;
constexpr Ring<Float> c{.x = 0.0, .y = 0.0, .r = 2.0, .w = L};

// =================================================================================================
namespace Expected {

constexpr std::array ns     = {16, 32, 64, 128, 256, 512, 1024};
constexpr std::array L1s_bc = {
    6.914268734335e-02,
    2.140080310971e-02,
    1.690965969347e-02,
    7.511566069029e-03,
    4.157310833304e-03,
    2.059576246615e-03,
    1.000232171813e-03,
};
constexpr std::array L1s_bl = {
    9.562740664038e-02,
    1.948384562686e-02,
    6.083316141843e-03,
    1.244389453128e-03,
    3.468374659071e-04,
    8.197649317376e-05,
    1.960065225332e-05,
};
constexpr std::array L1s_bq = {
    2.579999523437e-01,
    4.710281397358e-02,
    6.282780176066e-03,
    7.017925998244e-04,
    8.724374782096e-05,
    1.110287230971e-05,
    1.368174702030e-06,
};

constexpr auto L1_band(Index N) {
  switch (EXTRAPOLATION_ORDER) {
    case 0:  return interp_pow(ns, L1s_bc, N, 1);
    case 1:  return interp_pow(ns, L1s_bl, N, 2);
    case 2:  return interp_pow(ns, L1s_bq, N, 3);
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
  using namespace std::string_literals;
  switch (EXTRAPOLATION_ORDER) {
    case 0:  return "Constant"s;
    case 1:  return "Linear"s;
    case 2:  return "Quadratic"s;
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
    psi(i, j)    = c.levelset(x, y);
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

  struct Err {
    Float L1;
    Float L1_band;
    Float vol_band;
  };
  Err e = grid.transform_reduce_i(
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
  e.L1_band /= e.vol_band;

  Igor::Info("{} extrapolation", order2str());
  Igor::Info("L1({})      = {:.12e}", N, e.L1);
  Igor::Info("L1_band({}) = {:.12e}", N, e.L1_band);

  if (e.L1_band > 1.1 * Expected::L1_band(N)) {
    Igor::Error("Band error does not match expected value: expected {:.8e} but got {:.8e}",
                Expected::L1_band(N),
                e.L1_band);
    return 1;
  }
}
