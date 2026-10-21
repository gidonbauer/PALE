#pragma once

#include <Igor/Defer.hpp>

#include "BoundaryConditions.hpp"
#include "Grid.hpp"
#ifdef PALE_ASLAM_WENO
#include "WENO5.hpp"
#endif  // ALE_ASLAM_WENO

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void
normals(const Grid<Float, LAYOUT>& grid, const Scalar<Float, LAYOUT> psi, Vector<Float, LAYOUT> n) {
  PALE_PROFILE_FUNCTION;
  dispatch_metric(
      grid, DISPATCH_FUNC {
        grid.foreach_i(FOREACH_FUNC {
          const auto x        = grid.xm(i);
          const auto y        = grid.ym(j);
          const auto grad_psi = normalize(Vec2<Float>{
              .x = (psi(i + 1, j) - psi(i - 1, j)) / (2.0 * grid.dx() * Metric::h1(x, y)),
              .y = (psi(i, j + 1) - psi(i, j - 1)) / (2.0 * grid.dy() * Metric::h2(x, y)),
          });
          n.x(i, j)           = grad_psi.x;
          n.y(i, j)           = grad_psi.y;
        });
      });
}

namespace detail {

template <typename Float, Layout LAYOUT>
constexpr auto calc_tau_end(const Grid<Float, LAYOUT>& grid, Float band_width) -> Float {
  const auto factor = 4.0 * band_width;
  switch (grid.coords()) {
    case Coordinates::CARTESIAN:           return factor * std::max(grid.dx(), grid.dy());
    case Coordinates::SYMMETRIC_SPHERICAL:
    case Coordinates::POLAR: return factor * std::max(grid.r_max() * grid.dtheta(), grid.dr());
  }
  Igor::Panic("Unreachable");
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void advection_velocity(const Grid<Float, LAYOUT>& grid,
                                  const Scalar<Float, LAYOUT> psi,
                                  const Vector<Float, LAYOUT> n,
                                  Vector<Float, LAYOUT> u) {
  // u = H(psi)*n; where H(x) is the Heavyside function
  grid.foreach_i(FOREACH_FUNC {
    u.x(i, j) = psi(i, j) > 0.0 ? n.x(i, j) : 0.0;
    u.y(i, j) = psi(i, j) > 0.0 ? n.y(i, j) : 0.0;
  });
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Scalar<Float, LAYOUT>> SRC_t>
constexpr void aslam_advection_do_step_weno(const Grid<Float, LAYOUT>& grid,
                                            Float dt,
                                            const Scalar<Float, LAYOUT> f_old,
                                            const Vector<Float, LAYOUT> u,
                                            const FaceVector<Float, LAYOUT> fL,
                                            const FaceVector<Float, LAYOUT> fR,
                                            const SRC_t src,
                                            Scalar<Float, LAYOUT> f) {
  PALE_PROFILE_FUNCTION;
  dispatch_metric(
      grid, DISPATCH_FUNC {
        grid.foreach_i(FOREACH_FUNC {
          if (u.x(i, j) == 0.0 && u.y(i, j) == 0.0) { return; }

          const double f_minus = u.x(i, j) >= 0.0 ? fL.x(i, j) : fR.x(i, j);
          const double f_plus  = u.x(i, j) >= 0.0 ? fL.x(i + 1, j) : fR.x(i + 1, j);
          const double g_minus = u.y(i, j) >= 0.0 ? fL.y(i, j) : fR.y(i, j);
          const double g_plus  = u.y(i, j) >= 0.0 ? fL.y(i, j + 1) : fR.y(i, j + 1);

          const auto x         = grid.xm(i);
          const auto y         = grid.ym(j);
          f(i, j) =
              f_old(i, j) - dt * (u.x(i, j) * (f_plus - f_minus) / (grid.dx() * Metric::h1(x, y)) +
                                  u.y(i, j) * (g_plus - g_minus) / (grid.dy() * Metric::h2(x, y)));
          if constexpr (!IsNone<SRC_t>) { f(i, j) += dt * src(i, j); }
        });
      });
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Scalar<Float, LAYOUT>> SRC_t>
constexpr void aslam_advection_do_step_minmod(const Grid<Float, LAYOUT>& grid,
                                              Float dt,
                                              const Scalar<Float, LAYOUT> f_old,
                                              const Vector<Float, LAYOUT> u,
                                              const SRC_t src,
                                              Scalar<Float, LAYOUT> dfdt,
                                              Scalar<Float, LAYOUT> f) {
  PALE_PROFILE_FUNCTION;
  IGOR_ASSERT(grid.nghost() >= 2,
              "MinMod advection requires at least 2 ghost cells but got only {}",
              grid.nghost());

  auto aslam_minmod = [](double a, double b) {
    if (a * b <= 0.0) { return 0.0; }
    return std::abs(a) < std::abs(b) ? a : b;
  };

  dispatch_metric(
      grid, DISPATCH_FUNC {
        grid.foreach_i(FOREACH_FUNC {
          if (u.x(i, j) == 0.0 && u.y(i, j) == 0.0) { return; }

          const auto x = grid.xm(i);
          const auto y = grid.ym(j);
          // physical cell size
          const auto hx = grid.dx() * Metric::h1(x, y);
          const auto hy = grid.dy() * Metric::h2(x, y);

          dfdt(i, j)    = 0.0;
          // x-direction
          {
            // first differences
            const auto Dm = (f(i, j) - f(i - 1, j)) / hx;  // backward
            const auto Dp = (f(i + 1, j) - f(i, j)) / hx;  // forward

            // second differences (undivided) for the limiter
            const auto c_m = f(i, j) - 2.0 * f(i - 1, j) + f(i - 2, j);  // curvature left of i
            const auto c_c = f(i + 1, j) - 2.0 * f(i, j) + f(i - 1, j);  // curvature at i
            const auto c_p = f(i + 2, j) - 2.0 * f(i + 1, j) + f(i, j);  // curvature right of i

            // one-sided 2nd-order derivatives with MinMod curvature correction
            const auto fx_minus  = Dm + 0.5 * aslam_minmod(c_m, c_c) / hx;
            const auto fx_plus   = Dp - 0.5 * aslam_minmod(c_c, c_p) / hx;

            const auto fx        = u.x(i, j) >= 0.0 ? fx_minus : fx_plus;
            dfdt(i, j)          -= u.x(i, j) * fx;
          }

          // y-direction
          {
            // first differences
            const auto Dm = (f(i, j) - f(i, j - 1)) / hy;  // backward
            const auto Dp = (f(i, j + 1) - f(i, j)) / hy;  // forward

            // second differences (undivided) for the limiter
            const auto c_m = f(i, j) - 2.0 * f(i, j - 1) + f(i, j - 2);  // curvature left of i
            const auto c_c = f(i, j + 1) - 2.0 * f(i, j) + f(i, j - 1);  // curvature at i
            const auto c_p = f(i, j + 2) - 2.0 * f(i, j + 1) + f(i, j);  // curvature right of i

            // one-sided 2nd-order derivatives with MinMod curvature correction
            const auto fy_minus  = Dm + 0.5 * aslam_minmod(c_m, c_c) / hy;
            const auto fy_plus   = Dp - 0.5 * aslam_minmod(c_c, c_p) / hy;

            const auto fy        = u.y(i, j) >= 0.0 ? fy_minus : fy_plus;
            dfdt(i, j)          -= u.y(i, j) * fy;
          }

          if constexpr (!IsNone<SRC_t>) { dfdt(i, j) += src(i, j); }
        });
      });

  grid.foreach_i(FOREACH_FUNC { f(i, j) = f_old(i, j) + dt * dfdt(i, j); });
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_un(const Grid<Float, LAYOUT>& grid,
                       const Vector<Float, LAYOUT> n,
                       const Scalar<Float, LAYOUT> u,
                       const Scalar<Float, LAYOUT> psi,
                       Scalar<Float, LAYOUT> un) {
  dispatch_metric(
      grid, DISPATCH_FUNC {
        grid.foreach_i(FOREACH_FUNC {
          un(i, j) = 0.0;
          if (psi(i, j) <= 0.0) {
            const auto x = grid.xm(i);
            const auto y = grid.ym(j);

            // x-direction
            {
              if (psi(i + 1, j) <= 0.0 && psi(i - 1, j) <= 0.0) {
                // Central: -1/2 0 1/2
                un(i, j) +=
                    n.x(i, j) * (u(i + 1, j) - u(i - 1, j)) / (2.0 * grid.dx() * Metric::h1(x, y));
              } else if (psi(i + 1, j) <= 0.0 && psi(i + 2, j) <= 0.0) {
                // Forwards: −3/2 4/2 −1/2
                un(i, j) += n.x(i, j) * (-3.0 * u(i, j) + 4.0 * u(i + 1, j) - u(i + 2, j)) /
                            (2.0 * grid.dx() * Metric::h1(x, y));
              } else if (psi(i - 2, j) <= 0.0 && psi(i - 1, j) <= 0.0) {
                // Backwards: 1/2 −4/2 3/2
                un(i, j) += n.x(i, j) * (u(i - 2, j) - 4.0 * u(i - 1, j) + 3.0 * u(i, j)) /
                            (2.0 * grid.dx() * Metric::h1(x, y));
              } else {
                Igor::Panic("Neither central, forwards, or backwards finite differences are "
                            "possible, what now?");
              }
            }

            // y-direction
            {
              if (psi(i, j + 1) <= 0.0 && psi(i, j - 1) <= 0.0) {
                // Central: -1/2 0 1/2
                un(i, j) +=
                    n.y(i, j) * (u(i, j + 1) - u(i, j - 1)) / (2.0 * grid.dy() * Metric::h2(x, y));
              } else if (psi(i, j + 1) <= 0.0 && psi(i, j + 2) <= 0.0) {
                // Forwards: −3/2 4/2 −1/2
                un(i, j) += n.y(i, j) * (-3.0 * u(i, j) + 4.0 * u(i, j + 1) - u(i, j + 2)) /
                            (2.0 * grid.dy() * Metric::h2(x, y));
              } else if (psi(i, j - 2) <= 0.0 && psi(i, j - 1) <= 0.0) {
                // Backwards: 1/2 −4/2 3/2
                un(i, j) += n.y(i, j) * (u(i, j - 2) - 4.0 * u(i, j - 1) + 3.0 * u(i, j)) /
                            (2.0 * grid.dy() * Metric::h2(x, y));
              } else {
                Igor::Panic("Neither central, forwards, or backwards finite differences are "
                            "possible, what now?");
              }
            }
          }
        });
      });
}

}  // namespace detail

// =================================================================================================
template <typename Float,
          Layout LAYOUT,
          IsNoneOr<Vector<Float, LAYOUT>> N_t   = None,
          IsNoneOr<Scalar<Float, LAYOUT>> SRC_t = None>
constexpr void constant_extrapolate(const Grid<Float, LAYOUT>& grid,
                                    Scalar<Float, LAYOUT> f,
                                    Scalar<Float, LAYOUT> psi,
                                    Float band_width,
                                    N_t n_in  = None{},
                                    SRC_t src = None{}) {
  PALE_PROFILE_FUNCTION;

  auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
  IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
  if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

  auto u = grid.alloc_vector();
  IGOR_DEFER(grid.free(u););
  detail::advection_velocity(grid, psi, n, u);

  // Solve PDE
  // du/dtau + H(psi) n * grad(f) = 0

#ifdef PALE_ASLAM_WENO
  constexpr Float CFL = 0.9;
#else   // MinMod
  constexpr Float CFL = 0.4;
#endif  // PALE_ASLAM_WENO

  // Max. velocity is 1
  const Float dtau = CFL * [&] {
    switch (grid.coords()) {
      case Coordinates::CARTESIAN:           return std::min(grid.dx(), grid.dy());
      case Coordinates::SYMMETRIC_SPHERICAL:
      case Coordinates::POLAR:               return std::min(grid.rm(0) * grid.dtheta(), grid.dr());
    }
    Igor::Panic("Unreachable");
  }();

  auto f_old = grid.alloc_scalar();
  IGOR_DEFER(grid.free(f_old););
#ifdef PALE_ASLAM_WENO
  auto fL = grid.alloc_face_vector();
  auto fR = grid.alloc_face_vector();
  IGOR_DEFER(grid.free(fL););
  IGOR_DEFER(grid.free(fR););
#else
  auto dfdt = grid.alloc_scalar();
  IGOR_DEFER(grid.free(dfdt););
#endif  // PALE_ASLAM_WENO

  // Forward Euler step g <- g + dtau * L(g)
  const auto euler_step = [&](Scalar<Float, LAYOUT> g) {
#ifdef PALE_ASLAM_WENO
    // weno_reconstruction(grid, g, fL, fR);
    {
      IGOR_ASSERT(grid.nghost() >= 3, "WENO-5 scheme requires at least 3 ghost cells.");

      grid.template foreach_face_i<Dimension::X>(FOREACH_FUNC {
        if (!(psi(i - 1, j) >= 0.0 || psi(i, j) >= 0.0)) { return; }
        fL.x(i, j) = weno_left_biased_reconstruction(
            f(i - 3, j), f(i - 2, j), f(i - 1, j), f(i, j), f(i + 1, j));
        fR.x(i, j) = weno_right_biased_reconstruction(
            f(i - 2, j), f(i - 1, j), f(i, j), f(i + 1, j), f(i + 2, j));
      });

      grid.template foreach_face_i<Dimension::Y>(FOREACH_FUNC {
        if (!(psi(i, j - 1) >= 0.0 || psi(i, j) >= 0.0)) { return; }
        fL.y(i, j) = weno_left_biased_reconstruction(
            f(i, j - 3), f(i, j - 2), f(i, j - 1), f(i, j), f(i, j + 1));
        fR.y(i, j) = weno_right_biased_reconstruction(
            f(i, j - 2), f(i, j - 1), f(i, j), f(i, j + 1), f(i, j + 2));
      });
    }

    detail::aslam_advection_do_step_weno(grid, dtau, g, u, fL, fR, src, g);
#else
    detail::aslam_advection_do_step_minmod(grid, dtau, g, u, src, dfdt, g);
#endif
    apply_neumann_bconds(grid, g);
  };

  const auto tau_end = detail::calc_tau_end(grid, band_width);
  for (double tau = 0.0; tau <= tau_end; tau += dtau) {  // NOLINT
    copy(f, f_old);

    // = SSP-RK3 (Shu & Osher, 1988) =============
    // = First stage ====
    euler_step(f);

    // = Second stage ===
    euler_step(f);
    grid.foreach_i(FOREACH_FUNC { f(i, j) = 0.75 * f_old(i, j) + 0.25 * f(i, j); });
    apply_neumann_bconds(grid, f);

    // = Third stage ====
    euler_step(f);
    grid.foreach_i(FOREACH_FUNC { f(i, j) = f_old(i, j) / 3.0 + 2.0 / 3.0 * f(i, j); });
    apply_neumann_bconds(grid, f);
  }
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Vector<Float, LAYOUT>> N_t = None>
constexpr void linear_extrapolate(const Grid<Float, LAYOUT>& grid,
                                  Scalar<Float, LAYOUT> f,
                                  Scalar<Float, LAYOUT> psi,
                                  Float band_width,
                                  N_t n_in = None{}) {
  PALE_PROFILE_FUNCTION;

  auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
  IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
  if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

  // un = n*grad(u) if psi <= 0; 0 otherwise
  auto un = grid.alloc_scalar();
  IGOR_DEFER(grid.free(un););
  detail::calc_un(grid, n, f, psi, un);

  constant_extrapolate(grid, un, psi, band_width, n, None{});
  grid.foreach_i(FOREACH_FUNC { un(i, j) = psi(i, j) > 0.0 ? un(i, j) : 0.0; });

  constant_extrapolate(grid, f, psi, band_width, n, un);
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Vector<Float, LAYOUT>> N_t = None>
constexpr void quadratic_extrapolate(const Grid<Float, LAYOUT>& grid,
                                     Scalar<Float, LAYOUT> f,
                                     Scalar<Float, LAYOUT> psi,
                                     Float band_width,
                                     N_t n_in = None{}) {
  PALE_PROFILE_FUNCTION;

  auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
  IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
  if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

  // un = n*grad(u) if psi <= 0; 0 otherwise
  auto un = grid.alloc_scalar();
  IGOR_DEFER(grid.free(un););
  detail::calc_un(grid, n, f, psi, un);

  linear_extrapolate(grid, un, psi, band_width, n);
  grid.foreach_i(FOREACH_FUNC { un(i, j) = psi(i, j) > 0.0 ? un(i, j) : 0.0; });

  constant_extrapolate(grid, f, psi, band_width, n, un);
}

// =================================================================================================
template <Index ORDER = 1,
          typename Float,
          Layout LAYOUT,
          IsNoneOr<Vector<Float, LAYOUT>> N_t   = None,
          IsNoneOr<Scalar<Float, LAYOUT>> SRC_t = None>
constexpr void neumann_extrapolate(const Grid<Float, LAYOUT>& grid,
                                   Scalar<Float, LAYOUT> f,
                                   Scalar<Float, LAYOUT> psi,
                                   Float dfdn,
                                   Float band_width,
                                   N_t n_in = None{}) {
  PALE_PROFILE_FUNCTION;

  static_assert(ORDER == 1 || ORDER == 2,
                "ORDER can be 1 for linear or 2 for quadratic extrapolation.");

  if (ORDER == 1) {
    auto fn = grid.alloc_scalar();
    IGOR_DEFER(grid.free(fn););
    grid.foreach_i(FOREACH_FUNC { fn(i, j) = psi(i, j) > 0.0 ? dfdn : 0.0; });

    constant_extrapolate(grid, f, psi, band_width, n_in, fn);
  } else {
    auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
    IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
    if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

    auto fn  = grid.alloc_scalar();
    auto fnn = grid.alloc_scalar();
    IGOR_DEFER(grid.free(fn););
    IGOR_DEFER(grid.free(fnn););

    // Calculate fnn=n dot grad(n dot grad(f)) in the liquid and extrapolate constant
    detail::calc_un(grid, n, f, psi, fn);
    detail::calc_un(grid, n, fn, psi, fnn);
    constant_extrapolate(grid, fnn, psi, band_width, n, None{});

    // Calculate fn=dfdn + d*fnn. This works because fnn is constant along the normal
    grid.foreach_i(
        FOREACH_FUNC { fn(i, j) = psi(i, j) > 0.0 ? dfdn + fnn(i, j) * psi(i, j) : 0.0; });
    constant_extrapolate(grid, f, psi, band_width, n, fn);
  }
}
