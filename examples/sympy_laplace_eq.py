import numpy as np

from sympy import sin, cos
from sympy.abc import x, y

from context import poissonpy
from poissonpy import utils, solvers

interior = 0
left = utils.get_sp_function(sin(y))
right = utils.get_sp_function(sin(y))
top = utils.get_sp_function(sin(x))
bottom = utils.get_sp_function(sin(x))

boundary = {
    "left": (left, "dirichlet"),
    "right": (right, "dirichlet"),
    "top": (top, "dirichlet"),
    "bottom": (bottom, "dirichlet")
}

solver = solvers.Poisson2DRectangle(
    ((-2*np.pi, -2*np.pi), (2*np.pi, 2*np.pi)), interior, boundary, 100, 100)

solution = solver.solve()

utils.plot_3d(solver.x_grid, solver.y_grid, solution, "solution")
utils.plot_2d(solver.x, solver.y, solution, "solution")
#utils.plot_3d(solver.x_grid, solver.y_grid, gt, "ground truth")
