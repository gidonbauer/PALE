#pragma once

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "Grid.hpp"

// -------------------------------------------------------------------------------------------------
// template <typename Float>
// [[nodiscard]] constexpr auto intersect_line_line(const Vec2<Float>& a0,
//                                                  const Vec2<Float>& a1,
//                                                  const Vec2<Float>& b0,
//                                                  const Vec2<Float>& b1,
//                                                  Vec2<Float>& i) -> bool {
//   constexpr Float EPS = 1e-8;
//
//   const Float det     = (a1.x - a0.x) * (b0.y - b1.y) - (a1.y - a0.y) * (b0.x - b1.x);
//   if (std::abs(det) < EPS) { return false; }
//
//   const Float r = ((b0.y - b1.y) * (b0.x - a0.x) + (b1.x - b0.x) * (b0.y - a0.y)) / det;
//   const Float s = ((a0.y - a1.y) * (b0.x - a0.x) + (a1.x - a0.x) * (b0.y - a0.y)) / det;
//   if (!(0 - EPS <= r && r <= 1 + EPS) || !(0 - EPS <= s && s <= 1 + EPS)) { return false; }
//
//   i.x = a0.x + r * (a1.x - a0.x);
//   i.y = a0.y + r * (a1.y - a0.y);
//   return true;
// }

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

  // [[nodiscard]] constexpr auto intersect_line(Vec2<Float> p1, Vec2<Float> p2) const noexcept
  //     -> Vec2<Float> {
  //   auto sign           = [](Float x) -> Float { return x < 0 ? -1.0 : 1.0; };

  //   auto on_finite_line = [&](const Vec2<Float>& i) -> bool {
  //     constexpr Float EPS = 1e-8;
  //     return std::min(p1.x, p2.x) - EPS <= i.x && i.x <= std::max(p1.x, p2.x) + EPS &&
  //            std::min(p1.y, p2.y) - EPS <= i.y && i.y <= std::max(p1.y, p2.y) + EPS;
  //   };

  //   // See: https://mathworld.wolfram.com/Circle-LineIntersection.html
  //   p1.x                   -= x;
  //   p1.y                   -= y;
  //   p2.x                   -= x;
  //   p2.y                   -= y;

  //   const auto dx           = p2.x - p1.x;
  //   const auto dy           = p2.y - p1.y;
  //   const auto dr           = std::sqrt(dx * dx + dy * dy);
  //   const auto det          = p1.x * p2.y - p2.x * p1.y;

  //   const auto inside_sqrt  = r * r * dr * dr - det * det;
  //   if (!(inside_sqrt >= 0.0)) {
  //     Igor::Panic("Line ({:.6e}, {:.6e}) -- ({:.6e}, {:.6e}) and circle ({:.6e}, {:.6e},
  //     R={:.6e}) "
  //                 "do not intersect.",
  //                 p1.x + x,
  //                 p1.y + y,
  //                 p2.x + x,
  //                 p2.y + y,
  //                 x,
  //                 y,
  //                 r);
  //   }

  //   Vec2 i1 = {
  //       .x = (det * dy + sign(dy) * dx * std::sqrt(inside_sqrt)) / (dr * dr),
  //       .y = (-det * dx + std::abs(dy) * std::sqrt(inside_sqrt)) / (dr * dr),
  //   };
  //   Vec2 i2 = {
  //       .x = (det * dy - sign(dy) * dx * std::sqrt(inside_sqrt)) / (dr * dr),
  //       .y = (-det * dx - std::abs(dy) * std::sqrt(inside_sqrt)) / (dr * dr),
  //   };

  //   if (!(on_finite_line(i1) || on_finite_line(i2))) {
  //     Igor::Panic("None of the intersection points ({:.6e}, {:.6e}) and ({:.6e}, {:.6e}) is on
  //     the "
  //                 "finite line ({:.6e}, {:.6e}) -- ({:.6e}, {:.6e}).",
  //                 i1.x + x,
  //                 i1.y + y,
  //                 i2.x + x,
  //                 i2.y + y,
  //                 p1.x + x,
  //                 p1.y + y,
  //                 p2.x + x,
  //                 p2.y + y);
  //   }
  //   if (on_finite_line(i1) && on_finite_line(i2)) {
  //     Igor::Panic(
  //         "Both of the intersection points ({:.6e}, {:.6e}) and ({:.6e}, {:.6e}) are on the "
  //         "finite line ({:.6e}, {:.6e}) -- ({:.6e}, {:.6e}).",
  //         i1.x + x,
  //         i1.y + y,
  //         i2.x + x,
  //         i2.y + y,
  //         p1.x + x,
  //         p1.y + y,
  //         p2.x + x,
  //         p2.y + y);
  //   }

  //   if (on_finite_line(i1)) {
  //     i1.x += x;
  //     i1.y += y;
  //     return i1;
  //   } else {
  //     i2.x += x;
  //     i2.y += y;
  //     return i2;
  //   }
  // }
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

  // [[nodiscard]] constexpr auto intersect_line(const Vec2<Float>& p1,
  //                                             const Vec2<Float>& p2) const noexcept ->
  //                                             Vec2<Float> {
  //   std::array<Vec2<Float>, 4> intersects{};
  //   std::array<bool, 4> found_intersects{};

  //   found_intersects[0] =
  //       intersect_line_line(p1, p2, {.x = x, .y = y}, {.x = x + w, .y = y}, intersects[0]);
  //   found_intersects[1] =
  //       intersect_line_line(p1, p2, {.x = x, .y = y + h}, {.x = x + w, .y = y + h},
  //       intersects[1]);
  //   found_intersects[2] =
  //       intersect_line_line(p1, p2, {.x = x, .y = y}, {.x = x, .y = y + h}, intersects[2]);
  //   found_intersects[3] =
  //       intersect_line_line(p1, p2, {.x = x + w, .y = y}, {.x = x + w, .y = y + h},
  //       intersects[3]);

  //   int num_found = 0;
  //   for (auto found : found_intersects) {
  //     num_found += static_cast<int>(found);
  //   }
  //   if (num_found != 1) {
  //     Igor::Panic("Expected to find exactly one intersection but found {}", num_found);
  //   }

  //   for (size_t i = 0; i < intersects.size(); ++i) {
  //     if (found_intersects[i]) { return intersects[i]; }
  //   }

  //   Igor::Panic("Unreachable");
  //   std::unreachable();
  // }
};
