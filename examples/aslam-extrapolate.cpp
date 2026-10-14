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

// =================================================================================================
auto main(int argc, char** argv) -> int {
  const auto usage_str =
      Igor::detail::format("Usage: {} [-e=[constant|linear|quadratic]] <grid size>", argv[0]);

  enum class Method { CONSTANT, LINEAR, QUADRATIC };

  Index N       = 0;
  Method method = Method::CONSTANT;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.starts_with("-e")) {
      arg.remove_prefix(std::strlen("-e"));
      if (arg.starts_with('=')) {
        arg.remove_prefix(1);
      } else {
        if (i + 1 >= argc) {
          Igor::Error("Expected argument for `{}`", argv[i]);
          return 1;
        }
        i   += 1;
        arg  = argv[i];
      }
      if (arg == "constant" || arg == "const" || arg == "c") {
        method = Method::CONSTANT;
      } else if (arg == "linear" || arg == "l") {
        method = Method::LINEAR;
      } else if (arg == "quadratic" || arg == "q") {
        method = Method::QUADRATIC;
      } else {
        Igor::Error("{}", usage_str);
        Igor::Error("  Invalid extrapolation method `{}`", arg);
        return 1;
      }
    } else {
      if (std::from_chars(arg.data(), arg.data() + arg.size(), N).ec != std::errc{} || N <= 0) {
        Igor::Error("{}", usage_str);
        Igor::Error("  Invalid grid size `{}`", arg);
        return 1;
      }
    }
  }

  if (N == 0) {
    Igor::Error("{}", usage_str);
    Igor::Error("  Did not provide grid size.");
    return 1;
  }

  const auto output_dir = get_output_directory();
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
    switch (method) {
      case Method::CONSTANT:  u_exp(i, j) = u_expected_constant(x, y); break;
      case Method::LINEAR:    u_exp(i, j) = u_expected_linear(x, y); break;
      case Method::QUADRATIC: u_exp(i, j) = u_expected_quadratic(x, y); break;
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
    switch (method) {
      case Method::CONSTANT:  constant_extrapolate(grid, u, psi, tau_end); break;
      case Method::LINEAR:    linear_extrapolate(grid, u, psi, tau_end); break;
      case Method::QUADRATIC: quadratic_extrapolate(grid, u, psi, tau_end); break;
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

  switch (method) {
    case Method::CONSTANT:  Igor::Info("Constant extrapolation"); break;
    case Method::LINEAR:    Igor::Info("Linear extrapolation"); break;
    case Method::QUADRATIC: Igor::Info("Quadratic extrapolation"); break;
  }
  Igor::Info("L1({})      = {:.12e}", N, L1);
  Igor::Info("L1_band({}) = {:.12e}", N, L1_band);
}
