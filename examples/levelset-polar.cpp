#include <numbers>

#include "Geometry.hpp"
#include "Grid.hpp"
#include "HDFWriter.hpp"
#include "IO.hpp"

using Float       = double;
constexpr auto pi = std::numbers::pi_v<Float>;

auto main() -> int {
  const Index N = 128;
  Grid<Float> grid(0.0, 2.0 * pi, N, 0.0, 1.0, N, 3, Coordinates::POLAR);
  auto psi              = grid.alloc_scalar();

  const auto output_dir = get_output_directory();
  if (!init_output_directory(output_dir)) { return 1; }

  HDFWriter writer(output_dir, grid);
  writer.add_field("psi", psi);

  Float t = 0.0;

  {
    Circle<Float> shape{.x = 0.5, .y = 0.5, .r = 0.25};
    fill(psi, 0.0);
    grid.foreach_i(
        FOREACH_FUNC { psi(i, j) = shape.levelset(polar2cartesian(grid.xm(i), grid.ym(j))); });
    writer.write(t);
    t += 1.0;
  }

  {
    Ring<Float> shape{.x = 0.5, .y = 0.5, .r = 0.25, .w = 0.25};
    fill(psi, 0.0);
    grid.foreach_i(
        FOREACH_FUNC { psi(i, j) = shape.levelset(polar2cartesian(grid.xm(i), grid.ym(j))); });
    writer.write(t);
    t += 1.0;
  }

  {
    Ring<Float> shape{.x = 0.5, .y = 0.5, .r = 0.5, .w = -0.25};
    fill(psi, 0.0);
    grid.foreach_i(
        FOREACH_FUNC { psi(i, j) = shape.levelset(polar2cartesian(grid.xm(i), grid.ym(j))); });
    writer.write(t);
    t += 1.0;
  }

  {
    Rect<Float> shape{.x = 0.25, .y = 0.25, .w = 0.5, .h = 0.5};
    fill(psi, 0.0);
    grid.foreach_i(
        FOREACH_FUNC { psi(i, j) = shape.levelset(polar2cartesian(grid.xm(i), grid.ym(j))); });
    writer.write(t);
    t += 1.0;
  }

  {
    Rect<Float> shape{.x = 0.75, .y = 0.75, .w = -0.5, .h = -0.5};
    fill(psi, 0.0);
    grid.foreach_i(
        FOREACH_FUNC { psi(i, j) = shape.levelset(polar2cartesian(grid.xm(i), grid.ym(j))); });
    writer.write(t);
    t += 1.0;
  }
}
