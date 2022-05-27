import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from context import poissonpy
from poissonpy import functional, utils, solvers

source = np.array(Image.open("../data/poisson_image_editing/source.jpg")).astype(np.float64) / 255
mask = np.array(Image.open("../data/poisson_image_editing/mask.jpg").convert("L")).astype(np.float64) / 255
target = np.array(Image.open("../data/poisson_image_editing/target.jpg")).astype(np.float64) / 255

_, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

blended_rgb = []
# solve for each color channel
for i in range(source.shape[-1]):
    # compute laplacian of interpolation function
    Gx_src, Gy_src = functional.get_np_gradient(source[..., i])
    Gx_target, Gy_target = functional.get_np_gradient(target[..., i])
    G_src_mag = (Gx_src**2 + Gy_src**2)**0.5
    G_target_mag = (Gx_target**2 + Gy_target**2)**0.5
    Gx = np.where(G_src_mag > G_target_mag, Gx_src, Gx_target)
    Gy = np.where(G_src_mag > G_target_mag, Gy_src, Gy_target)
    Gxx, _ = functional.get_np_gradient(Gx, forward=False)
    _, Gyy = functional.get_np_gradient(Gy, forward=False)
    laplacian = Gxx + Gyy
    
    # solve interpolation function
    solver = solvers.Poisson2DRegion(mask, laplacian, target[..., i])
    solution = solver.solve()

    # alpha-blend interpolation and target function
    blended = mask * solution + (1 - mask) * target[..., i]
    blended_rgb.append(blended)

blended_rgb = np.dstack(blended_rgb)
blended_rgb = np.clip(blended_rgb, 0, 1)


fig, axes = plt.subplots(1, 3)
axes[0].imshow(source)
axes[1].imshow(target)
axes[2].imshow(blended_rgb)
axes[0].set_title("source")
axes[1].set_title("target")
axes[2].set_title("result")
axes[0].axis("off")
axes[1].axis("off")
axes[2].axis("off")
plt.tight_layout()
plt.show()