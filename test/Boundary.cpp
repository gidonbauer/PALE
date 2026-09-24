#include <cmath>
#include <limits>
#include <numbers>

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
// Upper bound on the number of reported mismatches per test; every mismatch still fails the test.
constexpr Index MAX_NUM_REPORTED = 10;

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

// Compare every entry of `s` for which `skip(i, j)` is false, ghost cells included, against
// `expected(i, j)`.
template <Layout LAYOUT, typename EXPECTED, typename SKIP>
auto check_all(std::string_view what, Scalar<Float, LAYOUT> s, EXPECTED expected, SKIP skip)
    -> bool {
  bool any_failed    = false;
  Index num_reported = 0;

  for (Index i = -s.nghost(); i < s.nx() + s.nghost(); ++i) {
    for (Index j = -s.nghost(); j < s.ny() + s.nghost(); ++j) {
      if (skip(i, j)) { continue; }
      const auto got  = s(i, j);
      const auto want = expected(i, j);
      if (!approx_eq(got, want)) {
        any_failed = true;
        if (num_reported < MAX_NUM_REPORTED) {
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

template <Layout LAYOUT, typename EXPECTED>
auto check_all(std::string_view what, Scalar<Float, LAYOUT> s, EXPECTED expected) -> bool {
  return check_all(what, s, expected, [](Index /*i*/, Index /*j*/) { return false; });
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
// = Neumann =======================================================================================
// =================================================================================================
// A zero normal gradient is imposed by mirroring the interior into the ghost cells. Where the
// mirror plane lies depends on the layout of the field:
//
//   - offset (cell centered): the boundary is the face between the indices -1 and 0, hence
//                             s(-k) = s(k - 1)  and  s(n - 1 + k) = s(n - k).
//   - align (boundary node):  the boundary is the node itself, hence
//                             s(-k) = s(k)      and  s(n - 1 + k) = s(n - 1 - k).
//
// On a boundary aligned field the value on the boundary is a degree of freedom of its own; it is
// set by the second order one-sided condition
//
//   ds/dx|_0 = (-3 * s(0) + 4 * s(1) - s(2)) / (2 * h) = 0   =>   s(0) = (4 * s(1) - s(2)) / 3,
//
// which `test_neumann_align_mirror` checks in its derivative form. That condition is exact only
// to O(h^3), so unlike a mirrored ghost cell the boundary value does not reproduce a smooth even
// function exactly and is excluded from `test_neumann_zero_gradient`.
//
constexpr auto offset_mirror_lo(Index k) -> Index { return k - 1; }
constexpr auto offset_mirror_hi(Index n, Index k) -> Index { return n - k; }
constexpr auto align_mirror_lo(Index k) -> Index { return k; }
constexpr auto align_mirror_hi(Index n, Index k) -> Index { return n - 1 - k; }

// A field that is even about all four boundaries: its normal derivative vanishes there, so a
// correct Neumann condition has to reproduce it exactly in the ghost cells.
auto even_about_boundaries(Float x, Float y) -> Float {
  constexpr auto pi = std::numbers::pi_v<Float>;
  return std::cos(pi * (x - X_MIN) / (X_MAX - X_MIN)) +
         std::cos(pi * (y - Y_MIN) / (Y_MAX - Y_MIN));
}

// Distinct values, so that a ghost cell identifies the interior cell it was copied from.
template <Layout LAYOUT>
void fill_unique(Scalar<Float, LAYOUT> s) {
  fill(s, std::numeric_limits<Float>::quiet_NaN());
  for (Index i = 0; i < s.nx(); ++i) {
    for (Index j = 0; j < s.ny(); ++j) {
      s(i, j) = static_cast<Float>(1 + i + s.nx() * j);
    }
  }
}

// `s(gi, gj)` has to be a copy of `s(si, sj)`; reports which cell it was copied from instead.
// `num_reported` caps the output, every mismatch still counts as a failure.
template <Layout LAYOUT>
auto check_mirror(std::string_view what,
                  Scalar<Float, LAYOUT> s,
                  Index gi,
                  Index gj,
                  Index si,
                  Index sj,
                  Index& num_reported) -> bool {
  const auto got = s(gi, gj);
  if (approx_eq(got, s(si, sj))) { return true; }

  if (num_reported < MAX_NUM_REPORTED) {
    num_reported += 1;
    if (std::isnan(got)) {
      Igor::Warn("{}: ({}, {}) was not filled, expected a copy of ({}, {}).", what, gi, gj, si, sj);
    } else {
      // Invert the fill to name the cell the value actually came from.
      const auto idx = static_cast<Index>(std::llround(got)) - 1;
      Igor::Warn("{}: ({}, {}) is a copy of ({}, {}) but has to be a copy of ({}, {}).",
                 what,
                 gi,
                 gj,
                 idx % s.nx(),
                 idx / s.nx(),
                 si,
                 sj);
    }
  }
  return false;
}

// =================================================================================================
// Cell centered scalar: mirrored about the boundary face.
auto test_neumann_offset_mirror() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  auto s = grid.alloc_scalar();
  fill_unique(s);

  // `clipped` is only implemented for the top boundary, the others abort; see
  // `test_neumann_clipped`.
  Neumann::apply_left_offset(grid, s);
  Neumann::apply_right_offset(grid, s);
  Neumann::apply_bottom_offset(grid, s);
  Neumann::apply_top_offset(grid, s);

  bool any_failed    = false;
  Index num_reported = 0;
  for (Index k = 1; k <= NGHOST; ++k) {
    for (Index j = 0; j < NY; ++j) {
      any_failed |= !check_mirror("left (offset)", s, -k, j, offset_mirror_lo(k), j, num_reported);
      any_failed |= !check_mirror(
          "right (offset)", s, NX - 1 + k, j, offset_mirror_hi(NX, k), j, num_reported);
    }
    for (Index i = 0; i < NX; ++i) {
      any_failed |=
          !check_mirror("bottom (offset)", s, i, -k, i, offset_mirror_lo(k), num_reported);
      any_failed |=
          !check_mirror("top (offset)", s, i, NY - 1 + k, i, offset_mirror_hi(NY, k), num_reported);
    }
  }

  return !any_failed;
}

// The second order one-sided derivative at the boundary node has to vanish:
// -3 * s0 + 4 * s1 - s2 == 0, where s1 and s2 are the first two interior neighbours.
auto check_zero_gradient_node(
    std::string_view what, Index at, Float s0, Float s1, Float s2, Index& num_reported) -> bool {
  const auto residual = -3.0 * s0 + 4.0 * s1 - s2;
  const auto scale    = 3.0 * std::abs(s0) + 4.0 * std::abs(s1) + std::abs(s2);
  if (std::abs(residual) <= REL_TOL * scale) { return true; }

  if (num_reported < MAX_NUM_REPORTED) {
    num_reported += 1;
    Igor::Warn("{}: at {}: (-3*{:.16e} + 4*{:.16e} - {:.16e}) = {:.8e}, has to be 0. The boundary "
               "value is {:.16e}, the one-sided condition gives {:.16e}.",
               what,
               at,
               s0,
               s1,
               s2,
               residual,
               s0,
               (4.0 * s1 - s2) / 3.0);
  }
  return false;
}

// =================================================================================================
// Boundary aligned field: has to be mirrored about the boundary node, and the value on the
// boundary node itself has to satisfy the one-sided zero gradient condition.
auto test_neumann_align_mirror() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  auto u = grid.alloc_face_vector();
  fill_unique(u.x);  // aligned in x-direction
  fill_unique(u.y);  // aligned in y-direction

  Neumann::apply_left_align(grid, u.x);
  Neumann::apply_right_align(grid, u.x);
  Neumann::apply_bottom_align(grid, u.y);
  Neumann::apply_top_align(grid, u.y);

  const Index unx    = u.x.nx();  // NX + 1 nodes, the boundary nodes are 0 and NX
  const Index uny    = u.y.ny();  // NY + 1 nodes, the boundary nodes are 0 and NY

  bool any_failed    = false;
  Index num_reported = 0;
  for (Index k = 1; k <= NGHOST; ++k) {
    for (Index j = 0; j < u.x.ny(); ++j) {
      any_failed |= !check_mirror("left (align)", u.x, -k, j, align_mirror_lo(k), j, num_reported);
      any_failed |= !check_mirror(
          "right (align)", u.x, unx - 1 + k, j, align_mirror_hi(unx, k), j, num_reported);
    }
    for (Index i = 0; i < u.y.nx(); ++i) {
      any_failed |=
          !check_mirror("bottom (align)", u.y, i, -k, i, align_mirror_lo(k), num_reported);
      any_failed |= !check_mirror(
          "top (align)", u.y, i, uny - 1 + k, i, align_mirror_hi(uny, k), num_reported);
    }
  }

  // - The value on the boundary node itself ------------------------------------
  Index num_node_reported = 0;
  for (Index j = 0; j < u.x.ny(); ++j) {
    any_failed |= !check_zero_gradient_node(
        "left (align) node", j, u.x(0, j), u.x(1, j), u.x(2, j), num_node_reported);
    any_failed |= !check_zero_gradient_node("right (align) node",
                                            j,
                                            u.x(unx - 1, j),
                                            u.x(unx - 2, j),
                                            u.x(unx - 3, j),
                                            num_node_reported);
  }
  for (Index i = 0; i < u.y.nx(); ++i) {
    any_failed |= !check_zero_gradient_node(
        "bottom (align) node", i, u.y(i, 0), u.y(i, 1), u.y(i, 2), num_node_reported);
    any_failed |= !check_zero_gradient_node("top (align) node",
                                            i,
                                            u.y(i, uny - 1),
                                            u.y(i, uny - 2),
                                            u.y(i, uny - 3),
                                            num_node_reported);
  }

  return !any_failed;
}

// =================================================================================================
// The same property stated physically: a field with a vanishing normal derivative has to be
// reproduced exactly in the ghost cells. Goes through the public entry points.
auto test_neumann_zero_gradient() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  const BConds<Float> bconds{
      .left   = Neumann{},
      .right  = Neumann{},
      .bottom = Neumann{},
      .top    = Neumann{},
  };
  bool any_failed = false;

  // - Cell centered scalar: offset in both directions --------------------------
  {
    auto s = make_scalar(grid, even_about_boundaries);
    apply_bconds(grid, bconds, s, 0.0);
    any_failed |= !check_all("apply_bconds", s, [&](Index i, Index j) {
      return even_about_boundaries(grid.xm(i), grid.ym(j));
    });
  }

  // - Face vector: aligned in the direction of the component -------------------
  // The values on the boundary nodes come from the one-sided condition, which is only accurate to
  // O(h^3), so they carry a truncation error; they are checked exactly in
  // `test_neumann_align_mirror`. Skipping them also skips the ghost cells of the other direction
  // that are mirrored from them.
  {
    auto u = make_u(grid, even_about_boundaries);
    apply_velocity_bconds(grid, bconds, bconds, u, 0.0);
    any_failed |= !check_all(
        "apply_velocity_bconds: u.x",
        u.x,
        [&](Index i, Index j) { return even_about_boundaries(grid.x(i), grid.ym(j)); },
        [&](Index i, Index /*j*/) { return i == 0 || i == u.x.nx() - 1; });
    any_failed |= !check_all(
        "apply_velocity_bconds: u.y",
        u.y,
        [&](Index i, Index j) { return even_about_boundaries(grid.xm(i), grid.y(j)); },
        [&](Index /*i*/, Index j) { return j == 0 || j == u.y.ny() - 1; });
  }

  return !any_failed;
}

// =================================================================================================
// `clipped` is only implemented for the top boundary: negative mirrored values become zero.
auto test_neumann_clipped() -> bool {
  Grid<Float> grid(X_MIN, X_MAX, NX, Y_MIN, Y_MAX, NY, NGHOST);
  auto s = grid.alloc_scalar();
  fill(s, std::numeric_limits<Float>::quiet_NaN());
  // Alternating signs, so that both branches are taken in every ghost layer.
  grid.foreach_i(FOREACH_FUNC { s(i, j) = ((i + j) % 2 == 0 ? 1.0 : -1.0) * (1.0 + j); });

  Neumann::apply_top_offset(grid, s, /*clipped=*/true);

  bool any_failed = false;
  for (Index k = 1; k <= NGHOST; ++k) {
    for (Index i = 0; i < NX; ++i) {
      const auto mirrored = s(i, offset_mirror_hi(NY, k));
      const auto expected = mirrored < 0.0 ? 0.0 : mirrored;
      if (!approx_eq(s(i, NY - 1 + k), expected)) {
        Igor::Warn("top (clipped): ({}, {}): got {:.16e}, expected {:.16e}",
                   i,
                   NY - 1 + k,
                   s(i, NY - 1 + k),
                   expected);
        any_failed = true;
      }
    }
  }

  return !any_failed;
}

// =================================================================================================
auto main() -> int {
  bool any_failed = false;

  // - Neumann ------------------------------------------------------------------
  if (!test_neumann_offset_mirror()) {
    Igor::Error("test_neumann_offset_mirror failed.");
    any_failed = true;
  }

  if (!test_neumann_align_mirror()) {
    Igor::Error("test_neumann_align_mirror failed.");
    any_failed = true;
  }

  if (!test_neumann_zero_gradient()) {
    Igor::Error("test_neumann_zero_gradient failed.");
    any_failed = true;
  }

  if (!test_neumann_clipped()) {
    Igor::Error("test_neumann_clipped failed.");
    any_failed = true;
  }

  // - Dirichlet ----------------------------------------------------------------
#ifdef PALE_BCONDS_LINEAR
  Igor::Warn("Built with `PALE_BCONDS_LINEAR`; the quadratic extrapolation is not tested.");
#else
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
#endif

  if (!any_failed) { Igor::Info("Boundary conditions: all tests passed."); }

  return any_failed ? 1 : 0;
}
