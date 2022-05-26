import numpy as np

import scipy.signal
from skimage.segmentation import find_boundaries

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from sympy import lambdify, diff
from sympy.abc import x, y

# Function related
def get_sp_function(expr):
    return lambdify([x, y], expr, "numpy")

def get_sp_laplacian_expr(expr):
    return diff(expr, x, 2) + diff(expr, y, 2)

def get_sp_derivative_expr(expr, var):
    return diff(expr, var, 1)
    
def get_np_gradient(arr, dx=1, dy=1, forward=True):
    if forward:
        kx = np.array([
            [0, 0, 0],
            [0, -1/dx, 1/dx],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, 0, 0],
            [0, -1/dy, 0],
            [0, 1/dy, 0]
        ])
    else:
        kx = np.array([
            [0, 0, 0],
            [-1/dx, 1/dx, 0],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, -1/dy, 0],
            [0, 1/dy, 0],
            [0, 0, 0]
        ])
    Gx = scipy.signal.fftconvolve(arr, kx, mode="same")
    Gy = scipy.signal.fftconvolve(arr, ky, mode="same")
    return Gx, Gy

def get_np_laplacian(arr, dx=1, dy=1):
    kernel = np.array([
        [0, 1/(dy**2), 0],
        [1/(dx**2), -2/(dx**2)-2/(dy**2), 1/(dx**2)],
        [0, 1/(dy**2), 0]
    ])
    laplacian = scipy.signal.fftconvolve(arr, kernel, mode="same")
    return laplacian

# Helper functions
def process_mask(mask):
    boundary = find_boundaries(mask, mode="inner").astype(int)
    inner = mask - boundary
    return inner, boundary

def get_grid_ids(X, Y):
    grid_ids = np.arange(Y * X).reshape(Y, X)
    return grid_ids

def get_selected_values(values, mask):
    assert values.shape == mask.shape
    nonzero_idx = np.nonzero(mask) # get mask 1
    return values[nonzero_idx]

# Plotting
def plot_2d(X, Y, Z, title):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(Z[::-1], cmap=cm.coolwarm)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks([0, len(X) - 1], np.round(np.array([X[0], X[-1]]), 2))
    ax.set_yticks([0, len(Y) - 1], np.round(np.array([Y[-1], Y[0]]), 2))
    fig.colorbar(heatmap, shrink=0.5, aspect=5)
    plt.show()

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
