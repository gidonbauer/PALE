#pragma once

#include <Igor/Defer.hpp>

#include "Grid.hpp"
#include "WENO5.hpp"

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void
normals(const Grid<Float, LAYOUT>& grid, const Scalar<Float, LAYOUT> psi, Vector<Float, LAYOUT> n) {
  grid.foreach_i(FOREACH_FUNC {
    const auto grad_psi = normalize(Vec2<Float>{
        .x = (psi(i + 1, j) - psi(i - 1, j)) / (2.0 * grid.dx()),
        .y = (psi(i, j + 1) - psi(i, j - 1)) / (2.0 * grid.dy()),
    });
    n.x(i, j)           = grad_psi.x;
    n.y(i, j)           = grad_psi.y;
  });
}

namespace detail {

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
constexpr void aslam_advection_do_step(const Grid<Float, LAYOUT>& grid,
                                       Float dt,
                                       const Scalar<Float, LAYOUT> f_old,
                                       const Vector<Float, LAYOUT> u,
                                       const FaceVector<Float, LAYOUT> fL,
                                       const FaceVector<Float, LAYOUT> fR,
                                       const SRC_t src,
                                       Scalar<Float, LAYOUT> f) {
  grid.foreach_i(FOREACH_FUNC {
    const double f_minus = u.x(i, j) >= 0.0 ? fL.x(i, j) : fR.x(i, j);
    const double f_plus  = u.x(i, j) >= 0.0 ? fL.x(i + 1, j) : fR.x(i + 1, j);
    const double g_minus = u.y(i, j) >= 0.0 ? fL.y(i, j) : fR.y(i, j);
    const double g_plus  = u.y(i, j) >= 0.0 ? fL.y(i, j + 1) : fR.y(i, j + 1);

    f(i, j)              = f_old(i, j) - dt * (u.x(i, j) * (f_plus - f_minus) / grid.dx() +
                                               u.y(i, j) * (g_plus - g_minus) / grid.dy());
    if constexpr (!IsNone<SRC_t>) { f(i, j) += dt * src(i, j); }
  });
}

// =================================================================================================
template <typename Float, Layout LAYOUT>
constexpr void calc_un(const Grid<Float, LAYOUT>& grid,
                       const Vector<Float, LAYOUT> n,
                       const Scalar<Float, LAYOUT> u,
                       const Scalar<Float, LAYOUT> psi,
                       Scalar<Float, LAYOUT> un) {
  grid.foreach_i(FOREACH_FUNC {
    un(i, j) = 0.0;
    if (psi(i, j) <= 0.0) {
      // x-direction
      {
        if (psi(i + 1, j) <= 0.0 && psi(i - 1, j) <= 0.0) {
          // Central: -1/2 0 1/2
          un(i, j) += n.x(i, j) * (u(i + 1, j) - u(i - 1, j)) / (2.0 * grid.dx());
        } else if (psi(i + 1, j) <= 0.0 && psi(i + 2, j) <= 0.0) {
          // Forwards: −3/2 4/2 −1/2
          un(i, j) +=
              n.x(i, j) * (-3.0 * u(i, j) + 4.0 * u(i + 1, j) - u(i + 2, j)) / (2.0 * grid.dx());
        } else if (psi(i - 2, j) <= 0.0 && psi(i - 1, j) <= 0.0) {
          // Backwards: 1/2 −4/2 3/2
          un(i, j) +=
              n.x(i, j) * (u(i - 2, j) - 4.0 * u(i - 1, j) + 3.0 * u(i, j)) / (2.0 * grid.dx());
        } else {
          Igor::Panic(
              "Neither central, forwards, or backwards finite differences are possible, what now?");
        }
      }

      // y-direction
      {
        if (psi(i, j + 1) <= 0.0 && psi(i, j - 1) <= 0.0) {
          // Central: -1/2 0 1/2
          un(i, j) += n.y(i, j) * (u(i, j + 1) - u(i, j - 1)) / (2.0 * grid.dy());
        } else if (psi(i, j + 1) <= 0.0 && psi(i, j + 2) <= 0.0) {
          // Forwards: −3/2 4/2 −1/2
          un(i, j) +=
              n.y(i, j) * (-3.0 * u(i, j) + 4.0 * u(i, j + 1) - u(i, j + 2)) / (2.0 * grid.dy());
        } else if (psi(i, j - 2) <= 0.0 && psi(i, j - 1) <= 0.0) {
          // Backwards: 1/2 −4/2 3/2
          un(i, j) +=
              n.y(i, j) * (u(i, j - 2) - 4.0 * u(i, j - 1) + 3.0 * u(i, j)) / (2.0 * grid.dy());
        } else {
          Igor::Panic(
              "Neither central, forwards, or backwards finite differences are possible, what now?");
        }
      }
    }
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
                                    Float tau_end = 0.1,
                                    N_t n_in      = None{},
                                    SRC_t src     = None{}) {
  auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
  IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
  if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

  auto u = grid.alloc_vector();
  IGOR_DEFER(grid.free(u););
  detail::advection_velocity(grid, psi, n, u);

  // Solve PDE
  // du/dtau + H(psi) n * grad(f) = 0
  constexpr Float CFL = 0.25;
  const Float dtau    = CFL * std::min(grid.dx(), grid.dy());  // Max. velocity is 1

  auto f_old          = grid.alloc_scalar();
  auto fL             = grid.alloc_face_vector();
  auto fR             = grid.alloc_face_vector();
  IGOR_DEFER(grid.free(f_old););
  IGOR_DEFER(grid.free(fL););
  IGOR_DEFER(grid.free(fR););
  for (double tau = 0.0; tau <= tau_end; tau += dtau) {  // NOLINT
    copy(f, f_old);

    // = Half step ===
    weno_reconstruction(grid, f, fL, fR);
    detail::aslam_advection_do_step(grid, 0.5 * dtau, f_old, u, fL, fR, src, f);

    // = Full step ===
    weno_reconstruction(grid, f, fL, fR);
    detail::aslam_advection_do_step(grid, dtau, f_old, u, fL, fR, src, f);
  }
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Vector<Float, LAYOUT>> N_t = None>
constexpr void linear_extrapolate(const Grid<Float, LAYOUT>& grid,
                                  Scalar<Float, LAYOUT> f,
                                  Scalar<Float, LAYOUT> psi,
                                  Float tau_end = 0.1,
                                  N_t n_in      = None{}) {
  auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
  IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
  if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

  // un = n*grad(u) if psi <= 0; 0 otherwise
  auto un = grid.alloc_scalar();
  IGOR_DEFER(grid.free(un););
  detail::calc_un(grid, n, f, psi, un);

  constant_extrapolate(grid, un, psi, tau_end, n, None{});
  grid.foreach_i(FOREACH_FUNC { un(i, j) = psi(i, j) > 0.0 ? un(i, j) : 0.0; });

  constant_extrapolate(grid, f, psi, tau_end, n, un);
}

// =================================================================================================
template <typename Float, Layout LAYOUT, IsNoneOr<Vector<Float, LAYOUT>> N_t = None>
constexpr void quadratic_extrapolate(const Grid<Float, LAYOUT>& grid,
                                     Scalar<Float, LAYOUT> f,
                                     Scalar<Float, LAYOUT> psi,
                                     Float tau_end = 0.1,
                                     N_t n_in      = None{}) {
  auto n = IF_NONE_ELSE(N_t, grid.alloc_vector(), n_in);
  IGOR_DEFER(if constexpr (IsNone<N_t>) { grid.free(n); });
  if constexpr (IsNone<N_t>) { normals(grid, psi, n); }

  // un = n*grad(u) if psi <= 0; 0 otherwise
  auto un = grid.alloc_scalar();
  IGOR_DEFER(grid.free(un););
  detail::calc_un(grid, n, f, psi, un);

  linear_extrapolate(grid, un, psi, tau_end, n);
  grid.foreach_i(FOREACH_FUNC { un(i, j) = psi(i, j) > 0.0 ? un(i, j) : 0.0; });

  constant_extrapolate(grid, f, psi, tau_end, n, un);
}
