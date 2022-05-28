import numpy as np
import matplotlib.pyplot as plt

from sympy import sin, cos
from sympy.abc import x, y

from context import poissonpy
from poissonpy import functional, utils, solvers


f_expr = sin(x**2 + y**2) / (x**2 + y**2)
laplacian_expr = functional.get_sp_laplacian_expr(f_expr)

f = functional.get_sp_function(f_expr)
laplacian = functional.get_sp_function(laplacian_expr)

interior = laplacian
boundary = f
mask = utils.circle_region(500, 500, 200)


solver = solvers.Poisson2DRegion(mask ,interior, boundary, ((-5, -5), (5, 5)))
gt = f(solver.x_grid, solver.y_grid) * mask

solution = solver.solve()
err = np.abs(gt - solution)

utils.plot_3d(solver.x_grid, solver.y_grid, gt, "ground truth")
utils.plot_3d(solver.x_grid, solver.y_grid, solution, "solution")
utils.plot_2d(solver.x, solver.y, solution, "solution")
utils.plot_2d(solver.x, solver.y, err, "error")