import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

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
