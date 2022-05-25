import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

from sympy import lambdify
from sympy.abc import x, y

def get_2d_sympy_function(expr):
    return lambdify([x, y], expr, "numpy")

def get_grid_ids(grid):
    grid_ids = np.arange(grid.shape[0] * grid.shape[1]).reshape(grid.shape[0], grid.shape[1])
    return grid_ids

def plot_3d(X, Y, Z, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title(title)
    plt.show()
