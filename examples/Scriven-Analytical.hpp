#pragma once

#include <cassert>
#include <cmath>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_gamma.h>

#include <Igor/Logging.hpp>

namespace Scriven {

struct Params {
  double Ja;
  double eps;
  double alpha;
  double Tsat;
  double Tinf;
  double beta;
  int dimension;
};

constexpr auto sq(double x) -> double { return x * x; }
constexpr auto cb(double x) -> double { return x * sq(x); }

constexpr auto integrad_3d(double x, void* params) -> double {
  const auto* p = static_cast<Params*>(params);
  return std::exp(-sq(x) - 2.0 * p->eps * cb(p->beta) / x) / sq(x);
}

constexpr auto g_3d(Params* p) -> double {
  return 2.0 * cb(p->beta) * std::exp((1.0 + 2.0 * p->eps) * sq(p->beta));
}

constexpr auto f_3d(double s, Params* p) -> double {
  gsl_integration_workspace* ws = gsl_integration_workspace_alloc(1000);

  gsl_function F                = {.function = integrad_3d, .params = p};
  double result;
  double abserr;
  gsl_integration_qagiu(&F,
                        s,
                        /*epsabs=*/0.0,
                        /*epsrel=*/1e-9,
                        /*limit=*/1000,
                        ws,
                        &result,
                        &abserr);
  // assert (abserr < result*1e-9);

  gsl_integration_workspace_free(ws);

  return result;
}

constexpr auto residual_2d(double beta, void* params) -> double {
  auto* p            = static_cast<Params*>(params);
  p->beta            = beta;
  const auto beta_sq = sq(beta);
  return p->Ja - std::pow(beta, 2.0 - 2.0 * p->eps * beta_sq) * std::exp(beta_sq) *
                     gsl_sf_gamma_inc(p->eps * beta_sq, beta_sq);
}

constexpr auto residual_3d(double beta, void* params) -> double {
  auto* p = static_cast<Params*>(params);
  p->beta = beta;
  return p->Ja - g_3d(p) * f_3d(beta, p);
}

void calc_beta(Params& p) {
  if (!(p.dimension == 2 || p.dimension == 3)) {
    Igor::Panic("Dimension {} is not supported.", p.dimension);
  }
  const auto residual = p.dimension == 2 ? residual_2d : residual_3d;

  gsl_function F      = {.function = residual, .params = &p};

  // Brent's method needs a bracket [x_lo, x_hi] that straddles the root.
  // Search outward from a small positive value if needed.
  double x_lo = 1e-4;
  double x_hi = 1.0;
  while (residual(x_lo, &p) * residual(x_hi, &p) > 0.0 && x_hi < 1e4) {
    x_hi *= 2.0;
  }
  assert(residual(x_lo, &p) * residual(x_hi, &p) < 0.0 && "calc_beta: could not bracket root");

  gsl_root_fsolver* solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
  gsl_root_fsolver_set(solver, &F, x_lo, x_hi);

  int status;
  int iter    = 0;
  double root = 0.0;
  do {
    ++iter;
    gsl_root_fsolver_iterate(solver);
    root   = gsl_root_fsolver_root(solver);
    x_lo   = gsl_root_fsolver_x_lower(solver);
    x_hi   = gsl_root_fsolver_x_upper(solver);
    status = gsl_root_test_interval(x_lo, x_hi, /*epsabs=*/0.0, /*epsrel=*/1e-9);
  } while (status == GSL_CONTINUE && iter < 200);
  assert(status == GSL_SUCCESS && "calc_beta: root finder did not converge");

  gsl_root_fsolver_free(solver);

  p.beta = root;
}

constexpr auto R(double t, const Params& p) -> double {
  return 2.0 * p.beta * std::sqrt(p.alpha * t);
}

constexpr auto t(double R, const Params& p) -> double { return sq(R / (2.0 * p.beta)) / p.alpha; }

constexpr auto T(double r, double t, Params& p) -> double {
  const double s = r / (2.0 * std::sqrt(p.alpha * t));
  switch (p.dimension) {
    case 2:
      {
        const auto beta_sq = sq(p.beta);
        return (p.Tsat - p.Tinf) * gsl_sf_gamma_inc(p.eps * beta_sq, sq(s)) /
                   gsl_sf_gamma_inc(p.eps * beta_sq, beta_sq) +
               p.Tinf;
      }
      break;
    case 3:  return p.Tinf - (p.Tinf - p.Tsat) / p.Ja * g_3d(&p) * f_3d(s, &p);
    default: Igor::Panic("Dimension {} is not supported.", p.dimension);
  }
}

constexpr auto beta_eff(double r, double r_dot, const Params& p) {
  return std::sqrt(0.5 * r * r_dot / p.alpha);
}

}  // namespace Scriven
