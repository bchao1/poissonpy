# poissonpy
Plug-and-play standalone library for solving 2D Poisson equations. Useful tool in scientific computing prototyping, image and video processing, computer graphics.

## Features
- Solves the Poisson equation on sqaure or non-square rectangular grids.
- Solves the Poisson equation on regions with arbitrary shape.
- Supports arbitrary boundary and interior conditions using `sympy` function experssions or `numpy` arrays.
- Supports Dirichlet, Neumann, or mixed boundary conditions.

## Disclaimer
This package is only used to solve 2D Poisson equations. If you are looking for a general purpose and optimized PDE library, you might want to checkout the [FEniCSx project](https://fenicsproject.org/index.html).

## Usage 
Import necessary libraries. `poissonpy` utilizes `numpy` and `sympy` greatly, so its best to import both:

```python
import numpy as np
from sympy import sin, cos
from sympy.abc import x, y

import poissonpy
```

### Compare with Analytical Solution
Define functions using `sympy` function expressions or `numpy` arrays:

```python
f_expr = sin(x) + cos(y)
laplacian_expr = diff(f_expr, x, 2) + diff(f_expr, y, 2)

f = poissonpy.get_2d_sympy_function(f_expr)
laplacian = poissonpy.get_2d_sympy_function(laplacian_expr)
```

Define interior and boundary conditions:

```python
interior = laplacian
boundary = {
    "left": (f, "dirichlet"),
    "right": (f, "dirichlet"),
    "top": (f, "dirichlet"),
    "bottom": (f, "dirichlet")
}
```

Initialize solver and solve Poisson equation:

```python
solver = Poisson2DRectangle(((-10, -5), (10, 5)), interior, boundary, X=100, Y=100)
solution = solver.solve()
```

Plot solution and ground truth:
```python
poissonpy.plot_3d(solver.x_grid, solver.y_grid, solution)
poissonpy.plot_3d(solver.x_grid, solver.y_grid, f(solver.x_grid, solver.y_grid))
```

|Solution|Ground truth|
|--|--|
|![](data/solution.png)|![](data/ground_truth.png)|

### Laplace Equation
It's also straightforward to define a Laplace equation - **we simply set the interior laplacian value to 0**. In the following example, we set the boundary values to be spatially-varying periodic functions.

```python
interior = 0 # laplace equation form
left = poissonpy.get_2d_sympy_function(sin(y))
right = poissonpy.get_2d_sympy_function(sin(y))
top = poissonpy.get_2d_sympy_function(sin(x))
bottom = poissonpy.get_2d_sympy_function(sin(x))

boundary = {
    "left": (left, "dirichlet"),
    "right": (right, "dirichlet"),
    "top": (top, "dirichlet"),
    "bottom": (bottom, "dirichlet")
}
```

Solve the Laplace equation:

```python
solver = Poisson2DRectangle(
    ((-2*np.pi, -2*np.pi), (2*np.pi, 2*np.pi)), interior, boundary, 100, 100)
solution = solver.solve()
poissonpy.plot_3d(solver.x_grid, solver.y_grid, solution, "solution")
poissonpy.plot_2d(solution, "solution")
```

|3D surface plot|2D heatmap|
|--|--|
|![](data/laplace_sol_3d.png)|![](data/laplace_sol_2d.png)|