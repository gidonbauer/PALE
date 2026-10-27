#pragma once

#include <cmath>

#include <Igor/Math.hpp>

template <typename Float>
struct Vec2 {
  Float x, y;

  [[nodiscard]] constexpr auto theta() noexcept -> Float& { return x; }
  [[nodiscard]] constexpr auto theta() const noexcept -> Float { return x; }
  [[nodiscard]] constexpr auto r() noexcept -> Float& { return y; }
  [[nodiscard]] constexpr auto r() const noexcept -> Float { return y; }
};

template <typename Float>
constexpr auto operator+(Vec2<Float> lhs, const Vec2<Float>& rhs) -> Vec2<Float> {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  return lhs;
}

template <typename Float>
constexpr auto operator*(Float lhs, Vec2<Float> rhs) -> Vec2<Float> {
  rhs.x *= lhs;
  rhs.y *= lhs;
  return rhs;
}

template <typename Float>
constexpr auto operator*(Vec2<Float> lhs, Float rhs) -> Vec2<Float> {
  return rhs * lhs;
}

template <typename Float>
constexpr auto operator/(Vec2<Float> lhs, Float rhs) -> Vec2<Float> {
  lhs.x /= rhs;
  lhs.y /= rhs;
  return lhs;
}

template <typename Float>
constexpr auto operator/=(Vec2<Float>& lhs, Float rhs) -> Vec2<Float> {
  lhs.x /= rhs;
  lhs.y /= rhs;
  return lhs;
}

template <typename Float>
[[nodiscard]] constexpr auto norm(const Vec2<Float>& vec) noexcept -> Float {
  return Igor::sqrt(Igor::sqr(vec.x) + Igor::sqr(vec.y));
}

template <typename Float>
[[nodiscard]] constexpr auto normalize(Vec2<Float> vec) noexcept -> Vec2<Float> {
  const Float n  = norm(vec);
  vec.x         /= n;
  vec.y         /= n;
  return vec;
}

// =================================================================================================
template <typename Float>
[[nodiscard]] constexpr auto cartesian2polar(const Vec2<Float>& p) -> Vec2<Float> {
  Vec2<Float> res{};
  res.theta() = std::atan2(p.y, p.x);
  res.r()     = norm(p);
  return res;
}
template <typename Float>
[[nodiscard]] constexpr auto cartesian2polar(Float x, Float y) -> Vec2<Float> {
  return cartesian2polar({.x = x, .y = y});
}

template <typename Float>
[[nodiscard]] constexpr auto polar2cartesian(const Vec2<Float>& p) -> Vec2<Float> {
  return Vec2<Float>{
      .x = p.r() * std::cos(p.theta()),
      .y = p.r() * std::sin(p.theta()),
  };
}
template <typename Float>
[[nodiscard]] constexpr auto polar2cartesian(Float theta, Float r) -> Vec2<Float> {
  Vec2<Float> p{};
  p.theta() = theta;
  p.r()     = r;
  return polar2cartesian(p);
}
