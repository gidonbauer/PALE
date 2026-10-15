#pragma once

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "Grid.hpp"

// levelset:
//   - => inside the liquid
//   0 => on the boundary
//   + => inside the solid

namespace detail {

template <typename Float>
constexpr auto levelset_width(Float xp, Float xb, Float w) -> Float {
  auto x_min = xb;
  auto x_max = xb + w;
  if (w < 0.0) { std::swap(x_min, x_max); }

  const auto d_min = xp - x_min;
  const auto d_max = x_max - xp;
  if (xp < x_min) { return d_min; }
  if (xp > x_max) { return d_max; }
  IGOR_ASSERT(
      d_min > 0.0 && d_max > 0.0, "Expected d_min={} and d_max={} to be <= 0.", d_min, d_max);
  return std::min(d_min, d_max);
}

}  // namespace detail

// -------------------------------------------------------------------------------------------------
template <typename Float>
struct Circle {
  Float x, y, r;

  [[nodiscard]] constexpr auto contains(Float x_, Float y_) const noexcept -> bool {
    return contains({.x = x_, .y = y_});
  }
  [[nodiscard]] constexpr auto contains(const Vec2<Float>& p) const noexcept -> bool {
    return Igor::sqr(p.x - x) + Igor::sqr(p.y - y) <= Igor::sqr(r);
  }

  [[nodiscard]] constexpr auto levelset(Float x_, Float y_) const noexcept -> Float {
    return levelset({.x = x_, .y = y_});
  }
  [[nodiscard]] constexpr auto levelset(Vec2<Float> p) const noexcept -> Float {
    return r - std::sqrt(Igor::sqr(p.x - x) + Igor::sqr(p.y - y));
  }
};

// -------------------------------------------------------------------------------------------------
template <typename Float>
struct Ring {
  Float x, y, r, w;

  [[nodiscard]] constexpr auto contains(Float x_, Float y_) const noexcept -> bool {
    return contains({.x = x_, .y = y_});
  }
  [[nodiscard]] constexpr auto contains(const Vec2<Float>& p) const noexcept -> bool {
    const auto d_sqr = Igor::sqr(p.x - x) + Igor::sqr(p.y - y);
    return Igor::sqr(r) <= d_sqr && d_sqr <= Igor::sqr(r + w);
  }

  [[nodiscard]] constexpr auto levelset(Float x_, Float y_) const noexcept -> Float {
    return levelset({.x = x_, .y = y_});
  }
  [[nodiscard]] constexpr auto levelset(Vec2<Float> p) const noexcept -> Float {
    const auto d = std::sqrt(Igor::sqr(p.x - x) + Igor::sqr(p.y - y));
    return detail::levelset_width(d, r, w);
  }
};

// -------------------------------------------------------------------------------------------------
template <typename Float>
struct Rect {
  Float x, y, w, h;

  [[nodiscard]] constexpr auto contains(Float x_, Float y_) const noexcept -> bool {
    return contains({.x = x_, .y = y_});
  }
  [[nodiscard]] constexpr auto contains(const Vec2<Float>& p) const noexcept -> bool {
    return x <= p.x && p.x <= x + w && y <= p.y && p.y <= y + h;
  }

  [[nodiscard]] constexpr auto levelset(Float x_, Float y_) const noexcept -> Float {
    return levelset({.x = x_, .y = y_});
  }
  [[nodiscard]] constexpr auto levelset(Vec2<Float> p) const noexcept -> Float {
    const Float d_x = detail::levelset_width(p.x, x, w);
    const Float d_y = detail::levelset_width(p.y, y, h);
    return std::min(d_x, d_y);
  }
};
