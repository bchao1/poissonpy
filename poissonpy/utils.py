import os
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Image utils
def read_image(filename, scale=1):
    img = Image.open(os.path.join(filename))
    if scale != 1:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    img = np.array(img)
    if len(img.shape) == 3:
        img = img[..., :3]
    return img.astype(np.float64) / 255 # only first 3

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def circle_region(X, Y, r):
    o_x, o_y = X // 2, Y // 2
    x_grid, y_grid = np.meshgrid(np.arange(X), np.arange(Y))
    circ = (((x_grid - o_x)**2 + (y_grid - o_y)**2) < r**2).astype(np.int)
    return circ

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
