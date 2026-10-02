# **P**olar **A**rbitrary **L**agrangian **E**ulerian (PALE) Solver for Simulation of Single Bubble Dynamics

The aim of this solver is to simulate the growth of a single bubble in an electrochemical system.
Only the liquid side is solved, the gas bubble is represented by the inner part of the polar mesh.
The growth of the bubble is implemented by moving the mesh using a simplified ALE method that only moves in radial direction.

## Quickstart

Build examples:
```
$ make
$ ./bin/<example>
```
Enable aggressive optimization and disable bound checks with `FAST=1`.
This should be done for every production run.
Build for parallel execution with `PARALLEL=1`.

Build and run tests:
```
$ ./test/test.py
```

Build benchmarks:
```
$ make FAST=1 bench
$ ./bin/bench/<benchmark>
```

Additional compile options are `DEBUG`, `SANITIZE`, `BACKTRACE` (requires [cpptrace](https://github.com/jeremy-rifkin/cpptrace)), and `SCOREP` (requires [Score-P](https://www.vi-hps.org/projects/score-p/overview/overview.html)).

## Numerical Method

The solver implements the prediction-projection method and uses 2nd order finite difference discretization on a MAC staggered mesh.
For the velocity and pressure, central finite differences are used.
The advection equation is discretized using the WENO5 scheme.
Time integration uses the 2nd order mid-point rule, i.e. explicit Runge-Kutta-2.

The solver supports Cartesian, polar, and rotationally symmetric spherical coordinates.
In the last case, the third dimension is assumed to be symmetric meaning $\frac{\partial (\cdot)}{\partial \phi} = 0$ and $u_{\phi} = 0$.
The solver does only 2D computation, with quasi-3D extension for the symmetric case.
The conventions for axis names are $x \equiv \theta$ and $y \equiv r$.
The names can be used interchangeably.

The coordinate systems are implemented using [orthogonal coordinates](https://en.wikipedia.org/wiki/Orthogonal_coordinates) and the respective metric tensor defined in `src/Metrics.hpp`.
More coordinate systems could be added in the same way.
Note that the multigrid solver then needs to be adjusted accordingly.

The ALE method supports only uniform movement in radial direction.
This simplifies the implementation, because the operators stay the same, only the volume-change needs to be accounted for.

The Poisson equation for the pressure is solved using a multigrid solver.
In the Cartesian case an over-relaxed red-black Gauss-Seidel smoother is used.
In the polar and spherical case zebra line relaxation along r is used.
This solves a linear system for an entire row in r-direction (y-direction) using the Thomas algorithm.
For parallelization every second row is skipped.

The solver supports Dirichlet (both constant value and a custom function), homogeneous Neumann and periodic boundary conditions.
Pressure can only have Neumann or periodic boundaries.
The multigrid solver makes the problem mean-free.
It subtracts the volume-weighted mean from the right-hand side, so pure-Neumann problems are solvable.

The solver uses $\Delta V = \Delta x \Delta y$ in the Cartesian case and $\Delta V = H \Delta \theta \Delta r$ in the polar and spherical case.
This means that in the spherical case each cell has unit depth in $\phi$ direction and to get the total mass one must multiply by $2 \pi$.

## Parallelization

The code is parallelized using the built-in parallelism of the C++ algorithms.
To make use of this, use the grid function `foreach[_face|_vertex]_[i|a]` or `transform_reduce[_face|_vertex]_[i|a]` or directly via the `range` variant of these functions.

## Output

Single values can be saved via the `Monitor` class. This generates a table with the value of a given variable for each time the write member function is called.
Whole fields can be saved using the `HDFWriter` class (preferred) or the `VTKWriter` class.

The output can be visualized with [ParaView](https://www.paraview.org/).

## Dependencies

- C++-23 compatible compiler, tested with `clang++`, `g++`, and `icpx`
- [Igor](https://github.com/gidonbauer/Igor): General utility. Path set via `IGOR_DIR`
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/): Output. Path set via `HDF_DIR`
- [PoisFFT](https://github.com/LadaF/PoisFFT): FFT-based Poisson solver. Vendored in `Thirdparty`. Path set via `POISFFT_DIR`
- [TBB](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html): Parallelization for `g++` and `icpx`. Path set via `TBB_DIR`
- [GSL](https://www.gnu.org/software/gsl/): Root finding in Scriven example. Path set via `GSL_DIR`
- [cpptrace](https://github.com/jeremy-rifkin/cpptrace): Optional for printing backtraces. Path set via `CPPTRACE_DIR`

## TODO

- [x] Look into toroidal coordinate systems to model the bubble at the wall: Does not look like a reasonable approach
    - Popov, Y.O., 2005. Evaporative deposition patterns: Spatial dimensions of the deposit. Phys. Rev. E 71, 036313. https://doi.org/10.1103/PhysRevE.71.036313
- [ ] Model the electrode via immersed boundaries

