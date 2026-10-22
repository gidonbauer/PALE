#include <charconv>
#include <numbers>

#include <Igor/Defer.hpp>
#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#define PALE_ASLAM_WENO
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

#ifndef POLAR_COORDINATES
// NOLINTNEXTLINE
#define POLAR_COORDINATES 0
#endif  // POLAR_COORDINATES

// =================================================================================================
using Float        = double;
constexpr Float pi = std::numbers::pi_v<Float>;

#if POLAR_COORDINATES
// Quarter annulus in polar coordinates. The liquid disk is off-center s.t. the normals are not
// aligned with the grid lines, and the solid contains cells with r*dtheta << dtheta.
constexpr Float theta_min = 0.0;
constexpr Float theta_max = pi / 2.0;
constexpr Float r_min     = 0.1;
constexpr Float r_max     = 3.0;
// Ring is wide enough that the levelset is the signed distance to the inner circle everywhere
constexpr Ring<Float> c{.x = 1.25, .y = 1.25, .r = 0.75, .w = 10.0};
#else
constexpr Float L = 2.0 * pi;
constexpr Ring<Float> c{.x = 0.0, .y = 0.0, .r = 2.0, .w = L};
#endif  // POLAR_COORDINATES

// =================================================================================================
namespace Expected {

constexpr std::array ns = {16, 32, 64, 128, 256, 512, 1024};

#ifdef PALE_ASLAM_WENO
constexpr Float factor = 1.0;
#else
constexpr Float factor = 1.3;
#endif
#if POLAR_COORDINATES
constexpr std::array L1s_bc = {
    factor * 4.109445114904e-02,
    factor * 1.189046212412e-02,
    factor * 6.366155488106e-03,
    factor * 3.012302252171e-03,
    factor * 1.751695037359e-03,
    factor * 7.439793913444e-04,
    factor * 3.699942822643e-04,
};
constexpr std::array L1s_bl = {
    factor * 4.295857313008e-02,
    factor * 1.170534120366e-02,
    factor * 2.359647583438e-03,
    factor * 6.034260609583e-04,
    factor * 1.559631166528e-04,
    factor * 3.569619095159e-05,
    factor * 8.672024898810e-06,
};
constexpr std::array L1s_bq = {
    factor * 3.535998107243e-02,
    factor * 8.913360962453e-03,
    factor * 1.459113048056e-03,
    factor * 1.686917214136e-04,
    factor * 2.165810812630e-05,
    factor * 2.568806935677e-06,
    factor * 3.220573091503e-07,
};
#else
constexpr std::array L1s_bc = {
    factor * 7.179318925943e-02,
    factor * 2.164714938758e-02,
    factor * 1.691828843162e-02,
    factor * 7.511870901688e-03,
    factor * 4.157243944131e-03,
    factor * 2.059499996582e-03,
    factor * 1.000190722218e-03,
};
constexpr std::array L1s_bl = {
    factor * 8.886804158077e-02,
    factor * 1.920951442053e-02,
    factor * 6.077255039370e-03,
    factor * 1.244388907551e-03,
    factor * 3.461840040656e-04,
    factor * 8.199221150557e-05,
    factor * 1.954571876051e-05,
};
constexpr std::array L1s_bq = {
    factor * 2.718003629114e-01,
    factor * 4.198600467317e-02,
    factor * 6.281530379979e-03,
    factor * 7.014206644087e-04,
    factor * 8.734944665125e-05,
    factor * 1.109675737865e-05,
    factor * 1.439671485376e-06,
};
#endif  // POLAR_COORDINATES

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
  return n.y * (temp4 * temp3 - temp2 * std::sin(y) * n.y) -
         n.x * (temp5 * temp1 + temp0 * std::cos(x) * n.x);
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

#if POLAR_COORDINATES
  const auto output_dir = "./test/output/Aslam-Polar-" + order2str() + '-' + std::to_string(N);
#else
  const auto output_dir = "./test/output/Aslam-" + order2str() + '-' + std::to_string(N);
#endif  // POLAR_COORDINATES
  if (!init_output_directory(output_dir)) { return 1; }

#if POLAR_COORDINATES
  Grid<Float> grid(theta_min, theta_max, N, r_min, r_max, N, 3, Coordinates::POLAR);
#else
  Grid<Float> grid(-L / 2.0, L / 2.0, N, -L / 2.0, L / 2.0, N, 3, Coordinates::CARTESIAN);
#endif  // POLAR_COORDINATES

  auto u     = grid.alloc_scalar();  // To-be-extrapolated value
  auto u_exp = grid.alloc_scalar();
  auto psi   = grid.alloc_scalar();  // Levelset

  // ===============================================================================================
  grid.foreach_a(FOREACH_FUNC {
#if POLAR_COORDINATES
    const auto [x, y] = polar2cartesian(grid.xm(i), grid.ym(j));
#else
    const auto x = grid.xm(i);
    const auto y = grid.ym(j);
#endif
    psi(i, j) = c.levelset(x, y);
    u(i, j)   = psi(i, j) > 0.0 ? 0.0 : u_func(x, y);
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

  // Band size and tau are physical lengths, use the largest physical cell size
  constexpr auto band_width = 3.0;
#if POLAR_COORDINATES
  const double band_size = band_width * std::max(grid.r_max() * grid.dtheta(), grid.dr());
#else
  const double band_size = band_width * std::max(grid.dx(), grid.dy());
#endif  // POLAR_COORDINATES

  IGOR_TIME_SCOPE("Extrapolation") {
    switch (EXTRAPOLATION_ORDER) {
      case 0:  constant_extrapolate(grid, u, psi, band_width); break;
      case 1:  linear_extrapolate(grid, u, psi, band_width); break;
      case 2:  quadratic_extrapolate(grid, u, psi, band_width); break;
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
