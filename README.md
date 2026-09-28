# **P**olar **A**rbitrary **L**agrangian **E**ulerian (PALE) Solver for Simulation Single Bubble Dynamics

The aim of this solver is to simulate the growth of a single bubble in an electrochemical system.
Only the liquid side is solved, the gas bubble is represented by inner part of the polar mesh.
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

The ALE method supports only uniform movement in radial direction.
This simplifies the implementation, because the operators stay the same, only the volume-change needs to be accounted for.

The Poisson equation for the pressure is solved using a multigrid solver.
In the Cartesian case a over-relaxed red-black Gauss-Seidel smoother is used.
In the polar case zebra line relaxation along r is used.
This solves a linear system for an entire row in r-direction (y-direction) using the Thomas algorithm.
For parallelization every second row is skipped.

## Parallelization

The code is parallelized using the built-in parallelism of the C++ algorithms.
To make use of this, use the grid function `foreach[_face|_vertex]_[i|a]` or `transform_reduce[_face|_vertex]_[i|a]` or directly via the `range` variant of these functions.

## Output

Single values can be saved via the `Monitor` class. This generates a table with the value of a given variable for each time the write member function is called.
Whole fields can be saved using the `HDFWriter` class (preferred) or the `VTKWriter` class.