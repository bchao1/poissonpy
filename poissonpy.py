import numpy as np
import sympy as sp
import scipy.sparse
import pyamg
import matplotlib.pyplot as plt
import seaborn as sns

import utils

# inner region conditions 

class Poisson2DRectangle:
    """ 
        Solve 2D Poisson Equation on a rectangle
    """
    def __init__(self, rect, interior, boundary, X=100, Y=100):
        """
        Args:
            rect:
            interior: sympy expresson or numpy array
            boundary:    
            N: Rectangle to N * N grid #
        """
        self.x1, self.y1 = rect[0] # top-left corner
        self.x2, self.y2 = rect[1] # bottom-right corner
        self.X, self.Y = X, Y

        x = np.linspace(self.x1, self.x2, self.X)
        y = np.linspace(self.y1, self.y2, self.Y)
        
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_grid, self.y_grid = np.meshgrid(x, y)
        self.xs = self.x_grid.flatten()
        self.ys = self.y_grid.flatten()

        self.interior = interior
        self.boundary = boundary
        

    def build_linear_system(self):

        interior_ids = np.arange(self.X + 1, 2 * self.X - 1) + self.X * np.expand_dims(np.arange(self.Y - 2), 1)
        interior_ids = interior_ids.flatten()
        boundary_ids = {
            "left": self.X * np.arange(1, self.Y - 1),
            "right": self.X * np.arange(1, self.Y - 1) + (self.X - 1),
            "top": np.arange(1, self.X - 1),
            "bottom": np.arange(self.X * self.Y - self.X + 1, self.X * self.Y - 1)
        }

        self.all_ids = np.concatenate([interior_ids, np.concatenate(list(boundary_ids.values()))]) 
        self.all_ids.sort()

        A = scipy.sparse.lil_matrix((len(self.all_ids), len(self.all_ids)))
        b = np.zeros(len(self.all_ids))

        interior_pos = np.searchsorted(self.all_ids, interior_ids)
        boundary_pos = {
            bd: np.searchsorted(self.all_ids, boundary_ids[bd]) for bd in boundary_ids
        }

        # interior - laplacian
        n1_pos = np.searchsorted(self.all_ids, interior_ids - 1)
        n2_pos = np.searchsorted(self.all_ids, interior_ids + 1)
        n3_pos = np.searchsorted(self.all_ids, interior_ids - self.X)
        n4_pos = np.searchsorted(self.all_ids, interior_ids + self.X)
        
        A[interior_pos, n1_pos] = 1 / (self.dx * self.dy) 
        A[interior_pos, n2_pos] = 1 / (self.dx * self.dy)
        A[interior_pos, n3_pos] = 1 / (self.dx * self.dy)
        A[interior_pos, n4_pos] = 1 / (self.dx * self.dy)
        A[interior_pos, interior_pos] = -4 / (self.dx * self.dy)
        b[interior_pos] = self.interior(self.xs[interior_ids], self.ys[interior_ids])


        for bd, (bd_func, mode) in self.boundary.items():
            bd_pos = boundary_pos[bd]
            bd_ids = boundary_ids[bd]
            b[bd_pos] = bd_func(self.xs[bd_ids], self.ys[bd_ids])
            if mode == "dirichlet":
                A[bd_pos, bd_pos] = 1
            elif mode == "neumann_x":
                if bd == "left":
                    n_ids = bd_ids + 1
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = -1 / self.dx
                    A[bd_pos, n_pos] = 1 / self.dx
                else: # right
                    n_ids = bd_ids - 1
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = 1 / self.dx
                    A[bd_pos, n_pos] = -1 / self.dx
            elif mode == "neumann_y":
                if bd == "top":
                    n_ids = bd_ids + self.X
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = -1 / self.dy
                    A[bd_pos, n_pos] = 1 / self.dy
                else: 
                    n_ids = bd_ids - self.X
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = 1 / self.dy
                    A[bd_pos, n_pos] = -1 / self.dy
        return A.tocsr(), b

    def solve(self):
        A, b = self.build_linear_system()
        x = scipy.sparse.linalg.spsolve(A, b)

        grid_ids = np.arange(self.Y * self.X)
        all_pos = np.searchsorted(grid_ids, self.all_ids)

        solution_grid = np.zeros(self.Y * self.X)
        solution_grid[all_pos] = x
        solution_grid = solution_grid.reshape(self.Y, self.X)
        return solution_grid


class Poisson2DRegion:
    """
        Solve 2D Poisson Equation on a region with arbitrary shape.
    """
    def __init__(self):
        pass

if __name__ == "__main__": 
    from sympy import lambdify, sin, cos, diff
    from sympy.abc import x, y

    f = sin(x) + cos(y)
    laplacian = diff(f, x, 2) + diff(f, y, 2)

    lambda_f = lambdify([x, y], f, "numpy")

    # analytic = sin(x) + cos(y)
    interior = lambdify([x, y], laplacian, "numpy")

    # possible boundary conditions: neumann_x, neumann_y, dirichlet

    left = lambdify([x, y], diff(f, x, 1), "numpy")
    right = lambdify([x, y], diff(f, x, 1), "numpy")
    top = lambdify([x, y], diff(f, y, 1), "numpy")
    bottom = lambdify([x, y], f, "numpy")

    boundary = {
        "left": (left, "neumann_x"),
        "right": (right, "neumann_x"),
        "top": (top, "neumann_y"),
        "bottom": (bottom, "dirichlet")
    }

    solver = Poisson2DRectangle(
        ((-6, -6), (6, 6)), interior, boundary, 100, 100)

    gt = lambda_f(solver.x_grid, solver.y_grid)

    solution = solver.solve()
    utils.plot_3d(solver.x_grid, solver.y_grid, solution)
    utils.plot_3d(solver.x_grid, solver.y_grid, gt)
    #sns.heatmap(gt)

    #plt.show()