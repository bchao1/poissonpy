import numpy as np

import scipy.signal

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

def get_np_div(gx, gy, dx=1, dy=1):
    gxx, _ = get_np_gradient(gx, dx, dy, forward=False)
    _, gyy = get_np_gradient(gy, dx, dy, forward=False)
    return gxx + gyy

def get_np_gradient_amplitude(gx, gy):
    return np.sqrt(gx**2 + gy**2)