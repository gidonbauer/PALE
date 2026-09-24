#include <cmath>
#include <limits>

#include <Igor/Logging.hpp>

#include "BoundaryConditions.hpp"
#include "Grid.hpp"

using Float            = double;

constexpr Index NGHOST = 3;
// Deliberately different in both directions to catch swapped x-/y-indices.
constexpr Index NX      = 6;
constexpr Index NY      = 5;
constexpr Float X_MIN   = -0.3;
constexpr Float X_MAX   = 1.7;
constexpr Float Y_MIN   = 0.4;
constexpr Float Y_MAX   = 2.9;

constexpr Float REL_TOL = 1e-12;

// =================================================================================================
// Separable quadratic; all coefficients are non-zero and distinct.
constexpr auto quad(Float x, Float y) -> Float {
  return 0.7 + 1.3 * x - 2.1 * x * x + 0.4 * y + 3.3 * y * y;
}
constexpr auto quad_left(Float y, Float /*t*/) -> Float { return quad(X_MIN, y); }
constexpr auto quad_right(Float y, Float /*t*/) -> Float { return quad(X_MAX, y); }
constexpr auto quad_bottom(Float x, Float /*t*/) -> Float { return quad(x, Y_MIN); }
constexpr auto quad_top(Float x, Float /*t*/) -> Float { return quad(x, Y_MAX); }

// Quadratics that depend on a single direction only; used to test the constant-value form of
// `Dirichlet::val`, for which the boundary value has to be constant along the boundary.
constexpr auto quad_x(Float x, Float /*y*/) -> Float { return 0.7 + 1.3 * x - 2.1 * x * x; }
constexpr auto quad_y(Float /*x*/, Float y) -> Float { return -0.9 + 0.4 * y + 3.3 * y * y; }

// A function that is not a polynomial; the identity checked in `test_wall_gradient` is algebraic
// and must hold for an arbitrary field.
auto non_polynomial(Float x, Float y) -> Float { return std::exp(0.8 * x) * std::sin(1.7 * y); }

// =================================================================================================
constexpr auto approx_eq(Float lhs, Float rhs, Float rel_tol = REL_TOL) -> bool {
  const auto scale = std::max({Float{1.0}, std::abs(lhs), std::abs(rhs)});
  return std::abs(lhs - rhs) <= rel_tol * scale;
}

// Compare every entry of `s`, ghost cells included, against `expected(i, j)`.
template <Layout LAYOUT, typename EXPECTED>
auto check_all(std::string_view what, Scalar<Float, LAYOUT> s, EXPECTED expected) -> bool {
  bool any_failed    = false;
  Index num_reported = 0;

  for (Index i = -s.nghost(); i < s.nx() + s.nghost(); ++i) {
    for (Index j = -s.nghost(); j < s.ny() + s.nghost(); ++j) {
      const auto got  = s(i, j);
      const auto want = expected(i, j);
      if (!approx_eq(got, want)) {
        any_failed = true;
        if (num_reported < 10) {
          Igor::Warn("{}: ({}, {}): got {:.16e}, expected {:.16e}, difference is {:.8e}",
                     what,
                     i,
                     j,
                     got,
                     want,
                     got - want);
          num_reported += 1;
        }
      }
    }
  }

  return !any_failed;
}

// =================================================================================================
// Cell centered scalar: offset in x- and in y-direction.
template <Layout LAYOUT>
auto make_scalar(const Grid<Float, LAYOUT>& grid, auto f) {
  auto s = grid.alloc_scalar();
  fill(s, std::numeric_limits<Float>::quiet_NaN());
  grid.foreach_i(FOREACH_FUNC { s(i, j) = f(grid.xm(i), grid.ym(j)); });
  return s;
}

// x-component of a face vector: aligned in x-, offset in y-direction.
template <Layout LAYOUT>
auto make_u(const Grid<Float, LAYOUT>& grid, auto f) {
  auto u = grid.alloc_face_vector();
  fill(u.x, std::numeric_limits<Float>::quiet_NaN());
  fill(u.y, std::numeric_limits<Float>::quiet_NaN());
  grid.template foreach_face_i<Dimension::X>(
      FOREACH_FUNC { u.x(i, j) = f(grid.x(i), grid.ym(j)); });
  grid.template foreach_face_i<Dimension::Y>(
      FOREACH_FUNC { u.y(i, j) = f(grid.xm(i), grid.y(j)); });
  return u;
}

// =================================================================================================
// The extrapolation has to be exact for a quadratic on every layout and on every side.
auto test_exact_for_quadratic() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  bool any_failed = false;

  // - Cell centered scalar: offset/offset -------------------------------------
  {
    auto s = make_scalar(grid, quad);
    Dirichlet<Float>::apply_left_offset(grid, s, true, 0.0, quad_left);
    Dirichlet<Float>::apply_right_offset(grid, s, true, 0.0, quad_right);
    Dirichlet<Float>::apply_bottom_offset(grid, s, true, 0.0, quad_bottom);
    Dirichlet<Float>::apply_top_offset(grid, s, true, 0.0, quad_top);

    any_failed |= !check_all("scalar (offset/offset)", s, [&](Index i, Index j) {
      return quad(grid.xm(i), grid.ym(j));
    });
  }

  // - Face vector: u.x is align/offset, u.y is offset/align --------------------
  {
    auto u = make_u(grid, quad);
    Dirichlet<Float>::apply_left_align(grid, u.x, true, 0.0, quad_left);
    Dirichlet<Float>::apply_right_align(grid, u.x, true, 0.0, quad_right);
    Dirichlet<Float>::apply_bottom_offset(grid, u.x, false, 0.0, quad_bottom);
    Dirichlet<Float>::apply_top_offset(grid, u.x, false, 0.0, quad_top);

    Dirichlet<Float>::apply_left_offset(grid, u.y, false, 0.0, quad_left);
    Dirichlet<Float>::apply_right_offset(grid, u.y, false, 0.0, quad_right);
    Dirichlet<Float>::apply_bottom_align(grid, u.y, true, 0.0, quad_bottom);
    Dirichlet<Float>::apply_top_align(grid, u.y, true, 0.0, quad_top);

    any_failed |= !check_all(
        "u.x (align/offset)", u.x, [&](Index i, Index j) { return quad(grid.x(i), grid.ym(j)); });
    any_failed |= !check_all(
        "u.y (offset/align)", u.y, [&](Index i, Index j) { return quad(grid.xm(i), grid.y(j)); });
  }

  return !any_failed;
}

// =================================================================================================
// The same, but going through the public entry points; catches a wrong `use_xm`/`use_ym` flag or a
// swapped align/offset variant in the dispatch.
auto test_apply_bconds() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  bool any_failed = false;

  const BConds<Float> bconds{
      .left   = Dirichlet<Float>{.val = quad_left},
      .right  = Dirichlet<Float>{.val = quad_right},
      .bottom = Dirichlet<Float>{.val = quad_bottom},
      .top    = Dirichlet<Float>{.val = quad_top},
  };

  {
    auto s = make_scalar(grid, quad);
    apply_bconds(grid, bconds, s, 0.0);
    any_failed |= !check_all(
        "apply_bconds", s, [&](Index i, Index j) { return quad(grid.xm(i), grid.ym(j)); });
  }

  {
    auto u = make_u(grid, quad);
    apply_velocity_bconds(grid, bconds, bconds, u, 0.0);
    any_failed |= !check_all("apply_velocity_bconds: u.x", u.x, [&](Index i, Index j) {
      return quad(grid.x(i), grid.ym(j));
    });
    any_failed |= !check_all("apply_velocity_bconds: u.y", u.y, [&](Index i, Index j) {
      return quad(grid.xm(i), grid.y(j));
    });
  }

  return !any_failed;
}

// =================================================================================================
// `Dirichlet::val` can also hold a constant instead of a function; that path has to extrapolate
// identically. A constant boundary value requires a field that is constant along the boundary,
// hence the two one-dimensional quadratics.
auto test_constant_value() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  bool any_failed = false;

  // - Varies in x-direction only: left and right have a constant boundary value -
  {
    auto s = make_scalar(grid, quad_x);
    Dirichlet<Float>::apply_left_offset(grid, s, true, 0.0, quad_x(X_MIN, 0.0));
    Dirichlet<Float>::apply_right_offset(grid, s, true, 0.0, quad_x(X_MAX, 0.0));

    // Only the left and right ghost cells are filled, the corners are not.
    for (Index i = -NGHOST; i < 0 && !any_failed; ++i) {
      for (Index j = 0; j < NY; ++j) {
        any_failed |= !approx_eq(s(i, j), quad_x(grid.xm(i), 0.0));
        any_failed |= !approx_eq(s(NX - 1 - i, j), quad_x(grid.xm(NX - 1 - i), 0.0));
      }
    }
    if (any_failed) { Igor::Warn("Constant value: left/right extrapolation is not exact."); }
  }

  // - Varies in y-direction only: bottom and top have a constant boundary value -
  {
    auto s = make_scalar(grid, quad_y);
    Dirichlet<Float>::apply_bottom_offset(grid, s, true, 0.0, quad_y(0.0, Y_MIN));
    Dirichlet<Float>::apply_top_offset(grid, s, true, 0.0, quad_y(0.0, Y_MAX));

    bool failed = false;
    for (Index j = -NGHOST; j < 0 && !failed; ++j) {
      for (Index i = 0; i < NX; ++i) {
        failed |= !approx_eq(s(i, j), quad_y(0.0, grid.ym(j)));
        failed |= !approx_eq(s(i, NY - 1 - j), quad_y(0.0, grid.ym(NY - 1 - j)));
      }
    }
    if (failed) { Igor::Warn("Constant value: bottom/top extrapolation is not exact."); }
    any_failed |= failed;
  }

  return !any_failed;
}

// =================================================================================================
// The property the Stefan problems rely on: the central difference across the boundary face that
// the flux calculation evaluates has to reduce to the second order one-sided difference
// (-8*v + 9*s0 - s1) / (3*h). This is an algebraic identity, so it must hold for any field.
auto test_wall_gradient() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  auto s = make_scalar(grid, non_polynomial);

  Dirichlet<Float>::apply_left_offset(
      grid, s, true, 0.0, +[](Float y, Float) { return non_polynomial(X_MIN, y); });
  Dirichlet<Float>::apply_right_offset(
      grid, s, true, 0.0, +[](Float y, Float) { return non_polynomial(X_MAX, y); });
  Dirichlet<Float>::apply_bottom_offset(
      grid, s, true, 0.0, +[](Float x, Float) { return non_polynomial(x, Y_MIN); });
  Dirichlet<Float>::apply_top_offset(
      grid, s, true, 0.0, +[](Float x, Float) { return non_polynomial(x, Y_MAX); });

  bool any_failed = false;

  for (Index j = 0; j < NY; ++j) {
    const auto v_left    = non_polynomial(X_MIN, grid.ym(j));
    const auto central   = (s(0, j) - s(-1, j)) / grid.dx();
    const auto one_sided = (-8.0 * v_left + 9.0 * s(0, j) - s(1, j)) / (3.0 * grid.dx());
    if (!approx_eq(central, one_sided)) {
      Igor::Warn("Left wall gradient at j={}: {:.16e} != {:.16e}", j, central, one_sided);
      any_failed = true;
    }

    const auto v_right   = non_polynomial(X_MAX, grid.ym(j));
    const auto central_r = (s(NX - 1, j) - s(NX, j)) / grid.dx();
    const auto one_sided_r =
        (-8.0 * v_right + 9.0 * s(NX - 1, j) - s(NX - 2, j)) / (3.0 * grid.dx());
    if (!approx_eq(central_r, one_sided_r)) {
      Igor::Warn("Right wall gradient at j={}: {:.16e} != {:.16e}", j, central_r, one_sided_r);
      any_failed = true;
    }
  }

  for (Index i = 0; i < NX; ++i) {
    const auto v_bottom  = non_polynomial(grid.xm(i), Y_MIN);
    const auto central   = (s(i, 0) - s(i, -1)) / grid.dy();
    const auto one_sided = (-8.0 * v_bottom + 9.0 * s(i, 0) - s(i, 1)) / (3.0 * grid.dy());
    if (!approx_eq(central, one_sided)) {
      Igor::Warn("Bottom wall gradient at i={}: {:.16e} != {:.16e}", i, central, one_sided);
      any_failed = true;
    }

    const auto v_top       = non_polynomial(grid.xm(i), Y_MAX);
    const auto central_t   = (s(i, NY - 1) - s(i, NY)) / grid.dy();
    const auto one_sided_t = (-8.0 * v_top + 9.0 * s(i, NY - 1) - s(i, NY - 2)) / (3.0 * grid.dy());
    if (!approx_eq(central_t, one_sided_t)) {
      Igor::Warn("Top wall gradient at i={}: {:.16e} != {:.16e}", i, central_t, one_sided_t);
      any_failed = true;
    }
  }

  return !any_failed;
}

// =================================================================================================
// The boundary value itself has to be recovered on a grid that is aligned with the boundary.
auto test_align_boundary_value() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  auto u          = make_u(grid, quad);
  bool any_failed = false;

  Dirichlet<Float>::apply_left_align(grid, u.x, true, 0.0, quad_left);
  Dirichlet<Float>::apply_right_align(grid, u.x, true, 0.0, quad_right);
  Dirichlet<Float>::apply_bottom_align(grid, u.y, true, 0.0, quad_bottom);
  Dirichlet<Float>::apply_top_align(grid, u.y, true, 0.0, quad_top);

  for (Index j = 0; j < NY; ++j) {
    if (!approx_eq(u.x(0, j), quad_left(grid.ym(j), 0.0))) {
      Igor::Warn("u.x(0, {}) is not the left boundary value.", j);
      any_failed = true;
    }
    if (!approx_eq(u.x(NX, j), quad_right(grid.ym(j), 0.0))) {
      Igor::Warn("u.x({}, {}) is not the right boundary value.", NX, j);
      any_failed = true;
    }
  }
  for (Index i = 0; i < NX; ++i) {
    if (!approx_eq(u.y(i, 0), quad_bottom(grid.xm(i), 0.0))) {
      Igor::Warn("u.y({}, 0) is not the bottom boundary value.", i);
      any_failed = true;
    }
    if (!approx_eq(u.y(i, NY), quad_top(grid.xm(i), 0.0))) {
      Igor::Warn("u.y({}, {}) is not the top boundary value.", i, NY);
      any_failed = true;
    }
  }

  return !any_failed;
}

// =================================================================================================
auto main() -> int {
#ifdef PALE_BCONDS_LINEAR
  Igor::Warn("Built with `PALE_BCONDS_LINEAR`; the quadratic extrapolation is not tested.");
  return 0;
#else
  bool any_failed = false;

  if (!test_exact_for_quadratic()) {
    Igor::Error("test_exact_for_quadratic failed.");
    any_failed = true;
  }

  if (!test_apply_bconds()) {
    Igor::Error("test_apply_bconds failed.");
    any_failed = true;
  }

  if (!test_constant_value()) {
    Igor::Error("test_constant_value failed.");
    any_failed = true;
  }

  if (!test_wall_gradient()) {
    Igor::Error("test_wall_gradient failed.");
    any_failed = true;
  }

  if (!test_align_boundary_value()) {
    Igor::Error("test_align_boundary_value failed.");
    any_failed = true;
  }

  if (!any_failed) { Igor::Info("Dirichlet boundary conditions: all tests passed."); }

  return any_failed ? 1 : 0;
#endif
}
