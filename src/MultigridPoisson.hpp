#pragma once

#include <Igor/Math.hpp>

#include "BoundaryConditions.hpp"
#include "Grid.hpp"
#include "Metrics.hpp"

template <typename Float, Layout LAYOUT>
class MultigridSolver {
  using Grid                               = Grid<Float, LAYOUT>;
  using Scalar                             = Scalar<Float, LAYOUT>;

  static constexpr Index MIN_NUM_ITER_POST = 2;
  static constexpr Index MAX_NUM_ITER_POST = 100;
  static constexpr Index MIN_NUM_ITER_PRE  = 1;
  static constexpr Index MAX_NUM_ITER_PRE  = 100;
  static constexpr Float OMEGA             = 1.2;

  struct Level {
    Grid grid;
    Scalar sol;
    Scalar rhs;
    Scalar res;

    // Restriction weights; normalized cell volumes of the four fine cells, separable in x and y
    std::vector<Float> restrict_wx_lo;  // (ic), fine cell 2*ic
    std::vector<Float> restrict_wx_hi;  // (ic), fine cell 2*ic + 1
    std::vector<Float> restrict_wy_lo;  // (jc), fine cell 2*jc
    std::vector<Float> restrict_wy_hi;  // (jc), fine cell 2*jc + 1

    // Separable cell volume, grid.dv(i, j) = vol_x(i) * vol_y(j)
    std::vector<Float> vol_x;  // (i)
    std::vector<Float> vol_y;  // (j)

    // Separable 5-point stencil of the Laplacian:
    //   L(i, j) = st_gx(j) * (st_x_lo(i) * s(i - 1, j) + st_x_hi(i) * s(i + 1, j))
    //           + st_y_lo(j) * s(i, j - 1) + st_y_hi(j) * s(i, j + 1) + st_diag(j) * s(i, j)
    std::vector<Float> st_x_lo;  // (i)
    std::vector<Float> st_x_hi;  // (i)
    std::vector<Float> st_gx;    // (j)
    std::vector<Float> st_y_lo;  // (j)
    std::vector<Float> st_y_hi;  // (j)
    std::vector<Float> st_diag;  // (j)

    // = Relevant only for polar and symmetric spherical coordinates ===========
    // Thomas coefficients for linear solver in r-direction
    std::vector<Float> tri_a;      // Sub-diagonal, zero at j = 0
    std::vector<Float> tri_cstar;  // Super-diagonal divided by the pivot
    std::vector<Float> tri_minv;   // Reciprocal of the pivot
    Float tri_bnd_lo = 0.0;  // Coupling to the ghost below, zero if it folds into the diagonal
    Float tri_bnd_hi = 0.0;  // Coupling to the ghost above, zero if it folds into the diagonal
    // = Relevant only for polar and symmetric spherical coordinates ===========
  };
  std::vector<Level> m_levels;
  BConds<Float> m_bconds;
  Index m_num_iter_pre;
  Index m_num_iter_post;
  Index m_num_cycles = 0;
  Float m_res        = 0.0;

  // -----------------------------------------------------------------------------------------------
  static constexpr void precompute_restriction_weights(const Level& fine, Level& coarse) {
    const auto calc_weights = [](const std::vector<Float>& vol,
                                 Index nc,
                                 std::vector<Float>& w_lo,
                                 std::vector<Float>& w_hi) {
      w_lo.resize(static_cast<size_t>(nc));
      w_hi.resize(static_cast<size_t>(nc));
      for (size_t c = 0; c < static_cast<size_t>(nc); ++c) {
        const Float v_lo = vol[2 * c];
        const Float v_hi = vol[2 * c + 1];
        w_lo[c]          = v_lo / (v_lo + v_hi);
        w_hi[c]          = v_hi / (v_lo + v_hi);
      }
    };
    calc_weights(fine.vol_x, coarse.grid.nx(), coarse.restrict_wx_lo, coarse.restrict_wx_hi);
    calc_weights(fine.vol_y, coarse.grid.ny(), coarse.restrict_wy_lo, coarse.restrict_wy_hi);
  }

  // -----------------------------------------------------------------------------------------------
  static constexpr void precompute_coefficients(Level& level, const BConds<Float>& bconds) {
    switch (level.grid.coords()) {
      case Coordinates::CARTESIAN:
        return precompute_coefficients_impl<Metric::Cartesian>(level, bconds);
      case Coordinates::POLAR: return precompute_coefficients_impl<Metric::Polar>(level, bconds);
      case Coordinates::SYMMETRIC_SPHERICAL:
        return precompute_coefficients_impl<Metric::SymmetricSpherical>(level, bconds);
    }
    Igor::Panic("Unreachable");
  }

  // Precompute the stencil, the cell volumes, and the coefficients for the Thomas algorithm in
  // r-direction (y-direction).
  //
  // Discretizes L = 1/H * [d/dq1(H/h1^2 * ds/dq1) + d/dq2(H/h2^2 * ds/dq2)] in conservative form.
  // Assumes that the metric is separable, i.e. h1 = h1(q2), h2 = h2(q2), and H = X(q1) * Y(q2).
  // This holds for Cartesian, polar, and symmetric spherical coordinates. Then the q1-coupling
  // factors as st_gx(j) * st_x_[lo|hi](i), and the q2-coupling depends only on j.
  template <typename Metric>
  static constexpr void precompute_coefficients_impl(Level& level, const BConds<Float>& bconds) {
    const Grid& grid    = level.grid;
    const Index nx      = grid.nx();
    const Index ny      = grid.ny();
    const Float inv_dx2 = 1.0 / Igor::sqr(grid.dx());
    const Float inv_dy2 = 1.0 / Igor::sqr(grid.dy());

    // Evaluate the factors of the separable metric at an arbitrary reference point
    const Float q1_ref = grid.xm(0);
    const Float q2_ref = grid.ym(0);

    level.vol_x.resize(static_cast<size_t>(nx));
    level.st_x_lo.resize(static_cast<size_t>(nx));
    level.st_x_hi.resize(static_cast<size_t>(nx));

    level.vol_y.resize(static_cast<size_t>(ny));
    level.st_gx.resize(static_cast<size_t>(ny));
    level.st_y_lo.resize(static_cast<size_t>(ny));
    level.st_y_hi.resize(static_cast<size_t>(ny));
    level.st_diag.resize(static_cast<size_t>(ny));

    level.tri_a.resize(static_cast<size_t>(ny));
    level.tri_cstar.resize(static_cast<size_t>(ny));
    level.tri_minv.resize(static_cast<size_t>(ny));

    // H/h1^2 at x-face, H/h2^2 at y-face
    const auto face_x = [](Float q1, Float q2) {
      return Metric::H(q1, q2) / Igor::sqr(Metric::h1(q1, q2));
    };
    const auto face_y = [](Float q1, Float q2) {
      return Metric::H(q1, q2) / Igor::sqr(Metric::h2(q1, q2));
    };

    // - q1-direction (x-direction) --------------------------------------------
    for (Index i = 0; i < nx; ++i) {
      const auto ii = static_cast<size_t>(i);
      // Undo the q2-dependence which is moved into st_gx
      const Float scale =
          Igor::sqr(Metric::h1(grid.xm(i), q2_ref)) / Metric::H(grid.xm(i), q2_ref) * inv_dx2;
      level.vol_x[ii]   = grid.dv(i, 0);
      level.st_x_lo[ii] = face_x(grid.x(i), q2_ref) * scale;
      level.st_x_hi[ii] = face_x(grid.x(i + 1), q2_ref) * scale;
    }

    // Sum of the q1-coefficients; independent of i for the supported metrics, e.g. for symmetric
    // spherical coordinates sin(theta - dtheta/2) + sin(theta + dtheta/2) =
    // 2 * cos(dtheta/2) * sin(theta). This allows a Thomas algorithm with coefficients only in j.
    const Float sum_x = level.st_x_lo[0] + level.st_x_hi[0];
    for (size_t ii = 0; ii < static_cast<size_t>(nx); ++ii) {
      IGOR_ASSERT(std::abs(level.st_x_lo[ii] + level.st_x_hi[ii] - sum_x) <= 1e-8 * sum_x,
                  "Diagonal of stencil depends on i: {} vs. {}",
                  level.st_x_lo[ii] + level.st_x_hi[ii],
                  sum_x);
    }

    // - q2-direction (y-direction) --------------------------------------------
    // Account for Neumann boundary conditions
    const bool fold_lo = std::holds_alternative<Neumann>(bconds.bottom);
    const bool fold_hi = std::holds_alternative<Neumann>(bconds.top);

    Float cstar_prev   = 0.0;
    for (Index j = 0; j < ny; ++j) {
      const auto jj     = static_cast<size_t>(j);
      const Float H_c   = Metric::H(q1_ref, grid.ym(j));

      level.vol_y[jj]   = grid.dv(0, j) / grid.dv(0, 0);
      level.st_gx[jj]   = 1.0 / Igor::sqr(Metric::h1(q1_ref, grid.ym(j)));
      level.st_y_lo[jj] = face_y(q1_ref, grid.y(j)) / H_c * inv_dy2;
      level.st_y_hi[jj] = face_y(q1_ref, grid.y(j + 1)) / H_c * inv_dy2;
      level.st_diag[jj] = -(level.st_y_lo[jj] + level.st_y_hi[jj]) - level.st_gx[jj] * sum_x;

      // Tridiagonal system in r-direction; q1-neighbours are moved to the right-hand side
      const Float a = level.st_y_lo[jj];  // Coefficient of lower diagonal
      const Float c = level.st_y_hi[jj];  // Coefficient of upper diagonal
      Float b       = level.st_diag[jj];  // Coefficient of diagonal

      if (j == 0) {
        if (fold_lo) {
          b += a;  // Adjustment for Neumann boundary condition
        } else {
          level.tri_bnd_lo = a;
        }
      }
      if (j == ny - 1) {
        if (fold_hi) {
          b += c;  // Adjustment for Neumann boundary condition
        } else {
          level.tri_bnd_hi = c;
        }
      }

      // The sub-diagonal is zero in first row
      const Float a_in = j == 0 ? 0.0 : a;
      // Pivot: b_j if j = 0, b_j - a_j*cstar_(j-1) otherwise
      const Float m       = b - a_in * cstar_prev;

      level.tri_a[jj]     = a_in;
      level.tri_minv[jj]  = 1.0 / m;
      level.tri_cstar[jj] = c / m;
      cstar_prev          = level.tri_cstar[jj];
    }
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void make_mean_free(const Level& level, Scalar s) const noexcept {
    struct SumVol {
      Float sum, vol;
    };
    const Float* vol_x = level.vol_x.data();
    const Float* vol_y = level.vol_y.data();
    const SumVol sv    = level.grid.transform_reduce_i(
        SumVol{.sum = 0.0, .vol = 0.0},
        FOREACH_FUNC {
          const Float dv = vol_x[i] * vol_y[j];
          return SumVol{.sum = s(i, j) * dv, .vol = dv};
        },
        [](SumVol lhs, const SumVol& rhs) {
          lhs.sum += rhs.sum;
          lhs.vol += rhs.vol;
          return lhs;
        });
    const auto mean = sv.sum / sv.vol;
    level.grid.foreach_i(FOREACH_FUNC { s(i, j) -= mean; });
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void residual(const Level& level) const noexcept {
    const Float inv_dx2 = 1.0 / Igor::sqr(level.grid.dx());
    const Float inv_dy2 = 1.0 / Igor::sqr(level.grid.dy());
    auto sol            = level.sol;
    auto rhs            = level.rhs;
    auto res            = level.res;

    apply_bconds(level.grid, m_bconds, sol, -1.0);
    switch (level.grid.coords()) {
      case Coordinates::CARTESIAN:
        level.grid.foreach_i(FOREACH_FUNC {
          const Float c = sol(i, j);
          const Float L = (sol(i - 1, j) - 2.0 * c + sol(i + 1, j)) * inv_dx2 +
                          (sol(i, j - 1) - 2.0 * c + sol(i, j + 1)) * inv_dy2;
          res(i, j)     = rhs(i, j) - L;
        });
        break;
      case Coordinates::POLAR:
      case Coordinates::SYMMETRIC_SPHERICAL:
        {
          // Get precomputed coefficients
          const Float* x_lo = level.st_x_lo.data();
          const Float* x_hi = level.st_x_hi.data();
          const Float* gx   = level.st_gx.data();
          const Float* y_lo = level.st_y_lo.data();
          const Float* y_hi = level.st_y_hi.data();
          const Float* diag = level.st_diag.data();

          level.grid.foreach_i(FOREACH_FUNC {
            const Float L = gx[j] * (x_lo[i] * sol(i - 1, j) + x_hi[i] * sol(i + 1, j)) +
                            y_lo[j] * sol(i, j - 1) + y_hi[j] * sol(i, j + 1) + diag[j] * sol(i, j);
            res(i, j)     = rhs(i, j) - L;
          });
        }
        break;
    }
  }

  // -----------------------------------------------------------------------------------------------
  constexpr auto max_res(const Level& level) const noexcept -> Float {
    const auto res = level.res;
    return level.grid.transform_reduce_i(
        0.0,
        FOREACH_FUNC { return std::abs(res(i, j)); },
        [](Float lhs, Float rhs) { return std::max(lhs, rhs); });
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void smooth_cartesian(const Level& level, Index num_iter) {
    const Float inv_dx2 = 1.0 / Igor::sqr(level.grid.dx());
    const Float inv_dy2 = 1.0 / Igor::sqr(level.grid.dy());
    const Float idiag   = 1.0 / (2.0 * (inv_dx2 + inv_dy2));
    const auto sol      = level.sol;
    const auto rhs      = level.rhs;
    const Index nx      = level.grid.nx();
    const Index ny      = level.grid.ny();

    for (Index iter = 0; iter < num_iter; ++iter) {
      // Red-black Gauss-Seidel with over-relaxation
      for (Index parity = 0; parity < 2; ++parity) {
        apply_bconds(level.grid, m_bconds, sol, -1.0);

        level.grid.foreach_range(0, nx, 0, (ny + 1) / 2, [=](Index i, Index jj) {
          const Index j = 2 * jj + (i + parity) % 2;
          if (j >= ny) { return; }

          sol(i, j) = (1.0 - OMEGA) * sol(i, j) +
                      OMEGA *
                          ((sol(i - 1, j) + sol(i + 1, j)) * inv_dx2 +
                           (sol(i, j - 1) + sol(i, j + 1)) * inv_dy2 - rhs(i, j)) *
                          idiag;
        });
      }
    }
  }

  // -----------------------------------------------------------------------------------------------
  // Zebra line relaxation along r; for polar and symmetric spherical coordinates
  constexpr void smooth_zebra(const Level& level, Index num_iter) {
    const auto sol     = level.sol;
    const auto rhs     = level.rhs;
    const Index nx     = level.grid.nx();
    const Index ny     = level.grid.ny();

    const Index sol_si = sol.stride_x();
    const Index sol_sj = sol.stride_y();
    const Index rhs_sj = rhs.stride_y();

    const Float* tri_a = level.tri_a.data();
    const Float* cstar = level.tri_cstar.data();
    const Float* minv  = level.tri_minv.data();
    const Float* x_lo  = level.st_x_lo.data();
    const Float* x_hi  = level.st_x_hi.data();
    const Float* gx    = level.st_gx.data();
    const Float bnd_lo = level.tri_bnd_lo;
    const Float bnd_hi = level.tri_bnd_hi;

    for (Index iter = 0; iter < num_iter; ++iter) {
      apply_bconds(level.grid, m_bconds, sol, -1.0);

      // Zebra line relaxation along r:
      // - Solve linear system for an entire row in r-direction (y-direction) -> Thomas algorithm
      // - Skip every second line for parallelization
      for (Index parity = 0; parity < 2; ++parity) {
        level.grid.foreach_range(0, (nx + 1 - parity) / 2, 0, 1, [=](Index ii, Index /*unused*/) {
          const Index i = 2 * ii + parity;

          // Use arrays for SIMD
          Float* sol_row       = sol.at(i, 0);      // Current row of sol; we solve for this
          const Float* row_lo  = sol_row - sol_si;  // Left/previous row
          const Float* row_hi  = sol_row + sol_si;  // Right/next row
          const Float* rhs_row = rhs.at(i, 0);      // Current row of rhs
          const Float c_lo     = x_lo[i];           // Coupling to left/previous row
          const Float c_hi     = x_hi[i];           // Coupling to right/next row

          // - Thomas algorithm ----------------------------
          // NOLINTBEGIN
          Float prev = 0.0;
          for (Index j = 0; j < ny; ++j) {
            // RHS for tridiagonal system; theta-derivative (x-derivative) is pulled to the right
            Float d = rhs_row[j * rhs_sj] -
                      gx[j] * (c_lo * row_lo[j * sol_sj] + c_hi * row_hi[j * sol_sj]);
            // Periodic boundary conditions
            if (j == 0) { d -= bnd_lo * sol_row[-sol_sj]; }
            if (j == ny - 1) { d -= bnd_hi * sol_row[ny * sol_sj]; }

            prev                = (d - tri_a[j] * prev) * minv[j];
            sol_row[j * sol_sj] = prev;
          }

          for (Index j = ny - 2; j >= 0; --j) {
            sol_row[j * sol_sj] -= cstar[j] * sol_row[(j + 1) * sol_sj];
          }
          // NOLINTEND
          // - Thomas algorithm ----------------------------
        });
      }
    }
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void smooth(const Level& level, Index num_iter) {
    // clang-format off
    switch (level.grid.coords()) {
      case Coordinates::CARTESIAN:           return smooth_cartesian(level, num_iter);  // Red-black GS
      case Coordinates::POLAR:               
      case Coordinates::SYMMETRIC_SPHERICAL: return smooth_zebra(level, num_iter);  // Zebra-line Thomas
    }
    // clang-format on
    Igor::Panic("Unreachable");
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void restrict_residual([[maybe_unused]] const Level& level,
                                   const Scalar fine_res,  // can be level.res or level.rhs
                                   const Level& coarse) {
    IGOR_ASSERT(level.grid.nx() / 2 == coarse.grid.nx() && level.grid.ny() / 2 == coarse.grid.ny(),
                "Expected `coarse` to be the next coarser level but we skipped something.");
    const auto res = fine_res;
    const auto rhs = coarse.rhs;

    // Residual of `level` becomes the rhs of `coarse`. Volume weighted average.
    const Float* wx_lo = coarse.restrict_wx_lo.data();
    const Float* wx_hi = coarse.restrict_wx_hi.data();
    const Float* wy_lo = coarse.restrict_wy_lo.data();
    const Float* wy_hi = coarse.restrict_wy_hi.data();

    coarse.grid.foreach_i(FOREACH_FUNC {
      rhs(i, j) =
          wy_lo[j] * (wx_lo[i] * res(2 * i, 2 * j) + wx_hi[i] * res(2 * i + 1, 2 * j)) +
          wy_hi[j] * (wx_lo[i] * res(2 * i, 2 * j + 1) + wx_hi[i] * res(2 * i + 1, 2 * j + 1));
    });
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void prolongate_and_correct(const Level& coarse, const Level& level) {
    IGOR_ASSERT(level.grid.nx() / 2 == coarse.grid.nx() && level.grid.ny() / 2 == coarse.grid.ny(),
                "Expected `coarse` to be the next coarser level but we skipped something.");
    const auto lsol = level.sol;
    const auto csol = coarse.sol;

    apply_bconds(coarse.grid, m_bconds, coarse.sol, -1.0);

    // Bilinear interpolation of the coarse correction onto the finer solution
    constexpr Float W = 1.0 / 16.0;
    coarse.grid.foreach_i(FOREACH_FUNC {
      const Float center = 9.0 * csol(i, j);
      const Float left   = 3.0 * csol(i - 1, j);
      const Float right  = 3.0 * csol(i + 1, j);
      const Float bottom = 3.0 * csol(i, j - 1);
      const Float top    = 3.0 * csol(i, j + 1);

      // clang-format off
      lsol(2 * i, 2 * j)         += (center + left  + bottom + csol(i - 1, j - 1)) * W;
      lsol(2 * i, 2 * j + 1)     += (center + left  + top    + csol(i - 1, j + 1)) * W;
      lsol(2 * i + 1, 2 * j)     += (center + right + bottom + csol(i + 1, j - 1)) * W;
      lsol(2 * i + 1, 2 * j + 1) += (center + right + top    + csol(i + 1, j + 1)) * W;
      // clang-format on
    });
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void vcycle(size_t l, bool sol_is_zero) {
    IGOR_ASSERT(l < m_levels.size(), "Level {} is out of bounds for {} levels", l, num_levels());
    Level& level = m_levels[l];

    // Check if we are at the coarsest level
    if (l + 1 == m_levels.size()) {
      smooth(level, m_num_iter_post);
      return;
    }

    Level& coarse = m_levels[l + 1];
    if (m_num_iter_pre > 0) {
      smooth(level, m_num_iter_pre);  // Remove high frequencies from residual
      residual(level);                // Iterate changed -> recompute residual
      sol_is_zero = false;
    }

    // Interpolate the residual of `level` onto `rhs` of coarse
    restrict_residual(level, sol_is_zero ? level.rhs : level.res, coarse);
    fill(coarse.sol, 0.0);
    vcycle(l + 1, true);  // Solve correction equation for `coarse`

    // Bilinear interpolation of the coarse correction onto level
    prolongate_and_correct(coarse, level);
    smooth(level, m_num_iter_post);
  }

 public:
  // -----------------------------------------------------------------------------------------------
  constexpr MultigridSolver(const Grid& grid,
                            BConds<Float> bconds =
                                {
                                    .left   = Neumann(),
                                    .right  = Neumann(),
                                    .bottom = Neumann(),
                                    .top    = Neumann(),
                                },
                            Index min_size      = 2,
                            Index num_iter_pre  = 0,
                            Index num_iter_post = 4)
      : m_bconds(std::move(bconds)),
        m_num_iter_pre(num_iter_pre),
        m_num_iter_post(num_iter_post) {
    IGOR_ASSERT(grid.nghost() >= 1, "Expected at least one ghost cell, but got {}", grid.nghost());
    IGOR_ASSERT(min_size >= 1, "Expected a positive minimum grid size, but got {}", min_size);

    Grid level_grid(grid.x_min(),
                    grid.x_max(),
                    grid.nx(),
                    grid.y_min(),
                    grid.y_max(),
                    grid.ny(),
                    grid.nghost(),
                    grid.coords());
    while (true) {
      m_levels.emplace_back(level_grid,
                            level_grid.alloc_scalar(),
                            level_grid.alloc_scalar(),
                            level_grid.alloc_scalar());
      precompute_coefficients(m_levels.back(), m_bconds);
      if (m_levels.size() > 1) {
        precompute_restriction_weights(m_levels[m_levels.size() - 2], m_levels.back());
      }

      if (level_grid.nx() % 2 != 0 || level_grid.ny() % 2 != 0 || level_grid.nx() / 2 < min_size ||
          level_grid.ny() / 2 < min_size) {
        break;
      }
      level_grid = Grid(level_grid.x_min(),
                        level_grid.x_max(),
                        level_grid.nx() / 2,
                        level_grid.y_min(),
                        level_grid.y_max(),
                        level_grid.ny() / 2,
                        level_grid.nghost(),
                        level_grid.coords());
    }
  }

  // -----------------------------------------------------------------------------------------------
  constexpr auto solve(Scalar sol, Scalar rhs, Float tol = 1e-4, Index max_iter = 100) -> bool {
    const Level& fine = m_levels[0];
    IGOR_ASSERT(sol.nx() == fine.sol.nx() && sol.ny() == fine.sol.ny() &&
                    sol.nghost() == fine.sol.nghost(),
                "Field `sol` does not match the grid the solver was constructed with.");
    IGOR_ASSERT(rhs.nx() == fine.rhs.nx() && rhs.ny() == fine.rhs.ny() &&
                    rhs.nghost() == fine.rhs.nghost(),
                "Field `rhs` does not match the grid the solver was constructed with.");

    copy(sol, fine.sol);
    copy(rhs, fine.rhs);
    make_mean_free(fine, fine.rhs);

    bool converged = false;

    residual(fine);
    m_res            = max_res(fine);
    Float res_before = m_res;
    for (m_num_cycles = 0; true; ++m_num_cycles) {
      if (m_res <= tol) {
        converged = true;
        break;
      }
      if (m_num_cycles == max_iter) { break; }

      vcycle(0, false);
      residual(fine);
      m_res = max_res(fine);

      // Dynamically adapt the number of smoothing iterations; adapted from Basilisk.
      // Pre- and post-smoothing move together.
      if (m_res > tol) {
        const Float reduction = res_before / m_res;
        if (reduction < 1.2) {
          if (m_num_iter_post < MAX_NUM_ITER_POST) { m_num_iter_post += 1; }
          if (m_num_iter_pre > 0 && m_num_iter_pre < MAX_NUM_ITER_PRE) { m_num_iter_pre += 1; }
        } else if (reduction > 10.0) {
          if (m_num_iter_post > MIN_NUM_ITER_POST) { m_num_iter_post -= 1; }
          if (m_num_iter_pre > MIN_NUM_ITER_PRE) { m_num_iter_pre -= 1; }
        }
      }
      res_before = m_res;
    }

    make_mean_free(fine, fine.sol);
    copy(fine.sol, sol);

    return converged;
  }

  // -----------------------------------------------------------------------------------------------
  constexpr void move_grid_by_velocity(const Vec2<Float>& w, Float dt) noexcept {
    for (size_t i = 0; i < m_levels.size(); ++i) {
      auto& level = m_levels[i];
      level.grid.move_grid_by_velocity(w, dt);
      precompute_coefficients(level, m_bconds);
      if (i > 0) {
        const auto& fine = m_levels[i - 1];
        precompute_restriction_weights(fine, level);
      }
    }
  }

  // -----------------------------------------------------------------------------------------------
  [[nodiscard]] constexpr auto num_levels() const noexcept -> Index {
    return static_cast<Index>(m_levels.size());
  }
  [[nodiscard]] constexpr auto num_cycles() const noexcept -> Index { return m_num_cycles; }
  [[nodiscard]] constexpr auto res() const noexcept -> Float { return m_res; }

  [[nodiscard]] constexpr auto num_iter_pre() noexcept -> Index& { return m_num_iter_pre; }
  [[nodiscard]] constexpr auto num_iter_pre() const noexcept -> Index { return m_num_iter_pre; }

  [[nodiscard]] constexpr auto num_iter_post() noexcept -> Index& { return m_num_iter_post; }
  [[nodiscard]] constexpr auto num_iter_post() const noexcept -> Index { return m_num_iter_post; }
};
