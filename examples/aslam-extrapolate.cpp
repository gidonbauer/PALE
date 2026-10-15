#include <charconv>
#include <fstream>
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
enum class Method { CONSTANT, LINEAR, QUADRATIC };
constexpr auto method2str(Method method) -> std::string {
  using namespace std::string_literals;
  switch (method) {
    case Method::CONSTANT:  return "Constant"s;
    case Method::LINEAR:    return "Linear"s;
    case Method::QUADRATIC: return "Quadratic"s;
  }
}

constexpr auto run(Method method, Index N, const std::string& base_output_dir, std::ostream& out)
    -> bool {
  const auto output_dir =
      base_output_dir + "/" + method2str(method) + "/" + std::to_string(N) + "/";
  if (!init_output_directory(output_dir)) { return false; }

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
  IGOR_TIME_SCOPE(method2str(method) + " extrapolation " + std::to_string(N)) {
    switch (method) {
      case Method::CONSTANT:  constant_extrapolate(grid, u, psi, tau_end); break;
      case Method::LINEAR:    linear_extrapolate(grid, u, psi, tau_end); break;
      case Method::QUADRATIC: quadratic_extrapolate(grid, u, psi, tau_end); break;
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

  out << method2str(method) << ','                                     //
      << N << ','                                                      //
      << std::setprecision(12) << std::scientific << e.L1 << ','       //
      << std::setprecision(12) << std::scientific << e.L1_band << ','  //
      << std::setprecision(12) << std::scientific << e.vol_band << '\n'
      << std::flush;
  return true;
}

// =================================================================================================
auto main(int argc, char** argv) -> int {
  const auto usage_str = Igor::detail::format("Usage: {} <max. level>", argv[0]);

  if (argc < 2) {
    Igor::Error("{}", usage_str);
    Igor::Error("  Did not provide max. level.");
    return 1;
  }

  Index max_level = 0;
  if (std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), max_level).ec != std::errc{} ||
      max_level <= 0) {
    Igor::Error("{}", usage_str);
    Igor::Error("  Invalid max. level `{}`", argv[1]);
    return 1;
  }

  const auto output_dir = get_output_directory();
  if (!init_output_directory(output_dir)) { return 1; }
  const auto scaling_filename = output_dir + "/scaling.csv";
  std::ofstream out(scaling_filename);
  if (!out) {
    Igor::Error("Could not open file `{}`: {}", scaling_filename, std::strerror(errno));
    return 1;
  }

  out << "method,N,L1,L1_band,vol_band\n";

  for (Method method : {Method::CONSTANT, Method::LINEAR, Method::QUADRATIC}) {
    for (Index level = 4; level <= max_level; ++level) {
      run(method, 1 << level, output_dir, out);
    }
  }
}
