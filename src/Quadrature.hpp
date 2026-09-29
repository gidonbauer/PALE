#pragma once

#include <numeric>
#include <span>

#include <Igor/Math.hpp>

#include "QuadratureTables.hpp"

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto quadrature(FUNC f, Float x_min, Float x_max) noexcept -> Float {
  static_assert(N > 0UZ && N <= detail::MAX_QUAD_N);

  constexpr auto& gauss_points  = detail::gauss_points_table<Float>[N - 1UZ];
  constexpr auto& gauss_weights = detail::gauss_weights_table<Float>[N - 1UZ];
  static_assert(gauss_points.size() == gauss_weights.size(),
                "Weights and points must have the same size.");
  static_assert(gauss_points.size() == N, "Number of weights and points must be equal to N.");
  static_assert(Igor::abs(std::reduce(gauss_weights.cbegin(), gauss_weights.cend()) -
                          static_cast<Float>(2)) <= 1e-15,
                "Weights must add up to 2.");

  auto integral = static_cast<Float>(0);
  for (size_t xidx = 0; xidx < gauss_points.size(); ++xidx) {
    const auto xi  = gauss_points[xidx];
    const auto w   = gauss_weights[xidx];

    const auto x   = (x_max - x_min) / 2 * xi + (x_max + x_min) / 2;
    integral      += w * f(x);
  }
  return (x_max - x_min) / 2 * integral;
}

// -------------------------------------------------------------------------------------------------
template <size_t N = 16UZ, typename FUNC, typename Float>
[[nodiscard]] constexpr auto
quadrature(FUNC f, Float x_min, Float x_max, Float y_min, Float y_max) noexcept -> Float {
  static_assert(N > 0UZ && N <= detail::MAX_QUAD_N);

  constexpr auto& gauss_points  = detail::gauss_points_table<Float>[N - 1UZ];
  constexpr auto& gauss_weights = detail::gauss_weights_table<Float>[N - 1UZ];
  static_assert(gauss_points.size() == gauss_weights.size(),
                "Weights and points must have the same size.");
  static_assert(gauss_points.size() == N, "Number of weights and points must be equal to N.");
  static_assert(Igor::abs(std::reduce(gauss_weights.cbegin(), gauss_weights.cend()) -
                          static_cast<Float>(2)) <= 1e-15,
                "Weights must add up to 2.");

  auto integral = static_cast<Float>(0);
  for (size_t xidx = 0; xidx < gauss_points.size(); ++xidx) {
    for (size_t yidx = 0; yidx < gauss_points.size(); ++yidx) {
      const auto wx    = gauss_weights[xidx];
      const auto wy    = gauss_weights[yidx];

      const auto xi_x  = gauss_points[xidx];
      const auto xi_y  = gauss_points[yidx];

      const auto x     = (x_max - x_min) / 2.0 * xi_x + (x_max + x_min) / 2.0;
      const auto y     = (y_max - y_min) / 2.0 * xi_y + (y_max + y_min) / 2.0;
      integral        += wx * wy * f(x, y);
    }
  }
  return (x_max - x_min) / 2.0 * (y_max - y_min) / 2.0 * integral;
}

// - Midpoint rule to integrate a function in 1D ---------------------------------------------------
template <typename Float>
[[nodiscard]] constexpr auto midpoint_rule(std::span<Float> f, Float dx) noexcept -> Float {
  return std::reduce(f.data(), f.data() + f.size(), static_cast<Float>(0)) * dx;
}

// - Trapezoidal rule to integrate a function in 1D ------------------------------------------------
template <typename Float>
[[nodiscard]] constexpr auto trapezoidal_rule(std::span<Float> f, std::span<Float> x) noexcept
    -> Float {
  IGOR_ASSERT(f.size() == x.size(),
              "f and x must have the same size, but f.size() = {} and x.size() = {}",
              f.size(),
              x.size());
  IGOR_ASSERT(f.size() >= 2UL, "Need at least two entries, but got {}", f.size());
  const auto N   = f.size();

  Float integral = 0.0;
  for (size_t i = 0; i < N - 1; ++i) {
    integral += (x[i + 1] - x[i]) * 0.5 * (f[i + 1] + f[i]);
  }
  return integral;
}

// - Simpson's rule to integrate a function in 1D --------------------------------------------------
template <typename Float>
[[nodiscard]] constexpr auto simpsons_rule(std::span<Float> f, Float x_min, Float x_max) noexcept
    -> Float {
  const auto N = f.size();
  IGOR_ASSERT(N > 0 && N % 2 == 1, "f.size() must be an odd number larger than zero but is {}", N);
  Float res = 0;
  for (size_t i = 1; i <= (N - 1) / 2; ++i) {
    res += f[2 * i - 2] + 4.0 * f[2 * i - 1] + f[2 * i];
  }
  const auto dx = (x_max - x_min) / static_cast<Float>(N - 1);
  return res * dx / 3.0;
}

// - Simpson's rule to integrate a function in 1D --------------------------------------------------
template <std::floating_point Float>
[[nodiscard]] constexpr auto simpson(std::span<const Float> y, std::span<const Float> x) -> Float {
  if (y.size() != x.size()) {
    Igor::Panic("`y` and `x` must have the same size, but sizes are {} and {}", y.size(), x.size());
  }

  const std::size_t n = y.size();
  if (n < 2) { return Float{0}; }
  if (n == 2) { return Float{0.5} * (x[1] - x[0]) * (y[0] + y[1]); }

  Float res = 0;

  // Composite Simpson over pairs of intervals [i, i+2], up to index `last`.
  const std::size_t last = (n % 2 == 1) ? n - 1 : n - 2;
  for (std::size_t i = 0; i + 2 <= last; i += 2) {
    const Float h0        = x[i + 1] - x[i];
    const Float h1        = x[i + 2] - x[i + 1];
    const Float hsum      = h0 + h1;
    const Float hprod     = h0 * h1;
    const Float h0_div_h1 = h0 / h1;
    res += hsum / Float{6} *
           (y[i] * (Float{2} - Float{1} / h0_div_h1) + y[i + 1] * (hsum * hsum / hprod) +
            y[i + 2] * (Float{2} - h0_div_h1));
  }

  // Cartwright correction for the final interval when N is even.
  if (n % 2 == 0) {
    const Float h0     = x[n - 2] - x[n - 3];
    const Float h1     = x[n - 1] - x[n - 2];
    const Float alpha  = (Float{2} * h1 * h1 + Float{3} * h0 * h1) / (Float{6} * (h0 + h1));
    const Float beta   = (h1 * h1 + Float{3} * h0 * h1) / (Float{6} * h0);
    const Float eta    = (h1 * h1 * h1) / (Float{6} * h0 * (h0 + h1));
    res               += alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3];
  }

  return res;
}
