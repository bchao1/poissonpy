import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from context import poissonpy
from poissonpy import functional, utils, solvers

ambient_rgb = utils.read_image("../data/flash_noflash/Objects_010_ambient.png", 0.25)
flash_rgb = utils.read_image("../data/flash_noflash/Objects_010_flash.png", 0.25)
mask = np.ones_like(ambient_rgb[..., 0])

ambient_rgb = np.pad(ambient_rgb, ((1, 1), (1, 1), (0, 0)), constant_values=1)
flash_rgb = np.pad(flash_rgb, ((1, 1), (1, 1), (0, 0)), constant_values=1)
mask = np.pad(mask, ((1, 1), (1, 1)))

res_rgb = []
for i in range(ambient_rgb.shape[-1]):
    ambient = ambient_rgb[..., i]
    flash = flash_rgb[..., i]

    gx_a, gy_a = functional.get_np_gradient(ambient)
    gx_f, gy_f = functional.get_np_gradient(flash)

    t = (gx_a*gx_f + gy_a*gy_f) / (gx_a**2 + gy_a**2 + 1e-8)
    gx_f_proj = t * gx_a
    gy_f_proj = t * gy_a

    lap = functional.get_np_div(gx_f_proj, gy_f_proj)
    solver = solvers.Poisson2DRegion(mask, lap, flash)
    res = solver.solve()
    res_rgb.append(res)
res_rgb = np.dstack(res_rgb)
res_rgb = np.clip(res_rgb, 0, 1)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(ambient_rgb)
axes[1].imshow(flash_rgb)
axes[2].imshow(res_rgb)
axes[0].set_title("ambient")
axes[1].set_title("flash")
axes[2].set_title("result")
axes[0].axis("off")
axes[1].axis("off")
axes[2].axis("off")
plt.tight_layout()
plt.show()
