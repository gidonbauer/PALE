#pragma once

#include <array>

#include "Grid.hpp"

namespace Expected {

// Interpolates values that behave like e = O(x^-p) by linearly interpolating
// C = e * x^p and returning C / n^p.
template <std::size_t N>
constexpr auto
interp_pow(const std::array<Index, N>& xs, const std::array<double, N>& es, Index n, unsigned p)
    -> double {
  static_assert(N >= 2, "need at least two samples");

  // Exact hits: return the tabulated value bit-for-bit.
  for (std::size_t i = 0; i < N; ++i) {
    if (xs[i] == n) { return es[i]; }
  }

  // Exponentiation by squaring; constexpr-friendly replacement for std::pow.
  // For p == 2 this computes exactly b * b, so results match interp_n2 bit-for-bit.
  const auto ipow = [p](double b) {
    double r = 1.0;
    for (unsigned k = p; k != 0U; k >>= 1U) {
      if ((k & 1U) != 0U) { r *= b; }
      b *= b;
    }
    return r;
  };

  const auto coeff = [&](std::size_t i) { return es[i] * ipow(static_cast<double>(xs[i])); };
  const auto np    = ipow(static_cast<double>(n));

  // Extrapolation: hold C at the nearest end -> pure power-law scaling.
  if (n < xs.front()) { return coeff(0) / np; }
  if (n > xs.back()) { return coeff(N - 1) / np; }

  // Bracket: xs[i] < n < xs[i + 1]
  std::size_t i = 0;
  while (xs[i + 1] < n) {
    ++i;
  }

  const auto x0 = static_cast<double>(xs[i]);
  const auto x1 = static_cast<double>(xs[i + 1]);
  const auto t  = (static_cast<double>(n) - x0) / (x1 - x0);

  return ((1.0 - t) * coeff(i) + t * coeff(i + 1)) / np;
}

template <std::size_t N>
constexpr auto interp_n2(const std::array<Index, N>& xs, const std::array<double, N>& es, Index n)
    -> double {
#if 1
  return interp_pow(xs, es, n, 2);
#else
  static_assert(N >= 2, "need at least two samples");

  // Exact hits: return the tabulated value bit-for-bit.
  for (std::size_t i = 0; i < N; ++i) {
    if (xs[i] == n) { return es[i]; }
  }

  const auto coeff = [&](std::size_t i) {
    const auto xi = static_cast<double>(xs[i]);
    return es[i] * xi * xi;
  };
  const auto nd = static_cast<double>(n);

  // Extrapolation: hold C at the nearest end -> pure quadratic scaling.
  if (n < xs.front()) { return coeff(0) / (nd * nd); }
  if (n > xs.back()) { return coeff(N - 1) / (nd * nd); }

  // Bracket: xs[i] < n < xs[i + 1]
  std::size_t i = 0;
  while (xs[i + 1] < n) {
    ++i;
  }

  const auto x0 = static_cast<double>(xs[i]);
  const auto x1 = static_cast<double>(xs[i + 1]);
  const auto t  = (nd - x0) / (x1 - x0);

  return ((1.0 - t) * coeff(i) + t * coeff(i + 1)) / (nd * nd);
#endif
}

}  // namespace Expected
