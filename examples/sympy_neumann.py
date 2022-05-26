import numpy as np

from sympy import sin, cos
from sympy.abc import x, y

from context import poissonpy
from poissonpy import functional, utils, solvers

    
# analytic = sin(x) + cos(y)
# laplacian = -sin(x) - cos(y)
# x derivative = cos(x)
# y derivative = -sin(y)
f_expr = sin(x) + cos(y)
laplacian_expr = functional.get_sp_laplacian_expr(f_expr)
x_derivative_expr = functional.get_sp_derivative_expr(f_expr, x)
y_derivative_expr = functional.get_sp_derivative_expr(f_expr, y)

f = functional.get_sp_function(f_expr)
laplacian = functional.get_sp_function(laplacian_expr)

# possible boundary conditions: neumann_x, neumann_y, dirichlet
interior = laplacian
left = functional.get_sp_function(x_derivative_expr)
right = f
top = functional.get_sp_function(y_derivative_expr)
bottom = f

boundary = {
    "left": (left, "neumann_x"),
    "right": (right, "dirichlet"),
    "top": (top, "neumann_y"),
    "bottom": (bottom, "dirichlet")
}

solver = solvers.Poisson2DRectangle(
    ((-2*np.pi, -2*np.pi), (2*np.pi, 2*np.pi)), interior, boundary, 100, 100)
gt = f(solver.x_grid, solver.y_grid)

solution = solver.solve()
err = np.abs(gt - solution)

utils.plot_2d(solver.x, solver.y, solution, "solution")
utils.plot_3d(solver.x_grid, solver.y_grid, solution, "solution")
utils.plot_3d(solver.x_grid, solver.y_grid, gt, "ground truth")
utils.plot_2d(solver.x, solver.y, err, "error")
