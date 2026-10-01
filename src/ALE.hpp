#pragma once

#include "Grid.hpp"

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void calc_H(const Grid<Float, LAYOUT>& grid, Scalar<Float, LAYOUT> H) {
  grid.foreach_i(FOREACH_FUNC { H(i, j) = Metric::H(grid.xm(i), grid.ym(j)); });
}

template <typename Float, Layout LAYOUT>
constexpr void calc_H(const Grid<Float, LAYOUT>& grid, Scalar<Float, LAYOUT> H) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:           return calc_H<Metric::Cartesian>(grid, H);
    case Coordinates::POLAR:               return calc_H<Metric::Polar>(grid, H);
    case Coordinates::SYMMETRIC_SPHERICAL: return calc_H<Metric::SymmetricSpherical>(grid, H);
  }
}

// =================================================================================================
template <typename Metric, typename Float, Layout LAYOUT>
constexpr void update_J(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Vec2<Float>& w,
                        const Scalar<Float, LAYOUT> J_old,
                        Scalar<Float, LAYOUT> J) {
  grid.foreach_i(FOREACH_FUNC {
    const auto H_left    = Metric::H(grid.x(i), grid.ym(j));
    const auto h1_left   = Metric::h1(grid.x(i), grid.ym(j));
    const auto HF_left   = H_left / h1_left * -w.x;

    const auto H_right   = Metric::H(grid.x(i + 1), grid.ym(j));
    const auto h1_right  = Metric::h1(grid.x(i + 1), grid.ym(j));
    const auto HF_right  = H_right / h1_right * -w.x;

    const auto dHFdx     = (HF_right - HF_left) / grid.dx();

    const auto H_bottom  = Metric::H(grid.xm(i), grid.y(j));
    const auto h2_bottom = Metric::h2(grid.xm(i), grid.y(j));
    const auto HF_bottom = H_bottom / h2_bottom * -w.y;

    const auto H_top     = Metric::H(grid.xm(i), grid.y(j + 1));
    const auto h2_top    = Metric::h2(grid.xm(i), grid.y(j + 1));
    const auto HF_top    = H_top / h2_top * -w.y;

    const auto dHFdy     = (HF_top - HF_bottom) / grid.dy();

    J(i, j)              = J_old(i, j) - dt * (dHFdx + dHFdy);
  });
}

template <typename Float, Layout LAYOUT>
constexpr void update_J(const Grid<Float, LAYOUT>& grid,
                        Float dt,
                        const Vec2<Float>& w,
                        const Scalar<Float, LAYOUT> J_old,
                        Scalar<Float, LAYOUT> J) {
  switch (grid.coords()) {
    case Coordinates::CARTESIAN: return update_J<Metric::Cartesian>(grid, dt, w, J_old, J);
    case Coordinates::POLAR:     return update_J<Metric::Polar>(grid, dt, w, J_old, J);
    case Coordinates::SYMMETRIC_SPHERICAL:
      return update_J<Metric::SymmetricSpherical>(grid, dt, w, J_old, J);
  }
}
