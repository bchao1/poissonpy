import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from context import poissonpy
from poissonpy import functional, utils, solvers

ambient_rgb = utils.read_image("../data/flash_noflash/Shelves_007_ambient.png", 0.25)
flash_rgb = utils.read_image("../data/flash_noflash/Shelves_007_flash.png", 0.25)
mask = np.ones_like(ambient_rgb[..., 0])

ambient_rgb = np.pad(ambient_rgb, ((1, 1), (1, 1), (0, 0)), constant_values=1)
flash_rgb = np.pad(flash_rgb, ((1, 1), (1, 1), (0, 0)), constant_values=1)
mask = np.pad(mask, ((1, 1), (1, 1)))

res_rgb = []
for i in range(ambient_rgb.shape[-1]):
    ambient = ambient_rgb[..., i]
    flash = flash_rgb[..., i]

    # compute image gradients
    gx_a, gy_a = functional.get_np_gradient(ambient)
    gx_f, gy_f = functional.get_np_gradient(flash)
    
    # gradient coherence map
    M = np.abs(gx_a * gx_f + gy_a * gy_f) / (functional.get_np_gradient_amplitude(gx_a, gy_a) * functional.get_np_gradient_amplitude(gx_f, gy_f) + 1e-8)
    reversal = (gx_a * gx_f + gy_a * gy_f) / (functional.get_np_gradient_amplitude(gx_a, gy_a) * functional.get_np_gradient_amplitude(gx_f, gy_f) + 1e-8) < 0
    # saturation map
    w_s = utils.normalize(np.tanh(40 * (utils.normalize(flash) - 0.9)))

    t = np.abs(gx_a*gx_f + gy_a*gy_f) / (gx_f**2 + gy_f**2)
    gx_a_proj = t * gx_f
    gy_a_proj = t * gy_f
    gx_a_proj = np.where(reversal, -gx_a_proj, gx_a_proj)
    gy_a_proj = np.where(reversal, -gy_a_proj, gy_a_proj)
    #plt.imshow(gx_a_proj, cmap="gray")
    #plt.show()

    gx_a_new = gx_a_proj#w_s * gx_a + (1 - w_s) * (gx_a_proj)
    gy_a_new = gy_a_proj#w_s * gy_a + (1 - w_s) * (gy_a_proj)
    #gx_f_new = w_s * gx_a + (1 - w_s) * (M * gx_f + (1 - M) * gx_a)
    #gy_f_new = w_s * gy_a + (1 - w_s) * (M * gy_f + (1 - M) * gy_a)

    lap = functional.get_np_div(gx_a_new, gy_a_new)

    solver = solvers.Poisson2DRegion(mask, lap, ambient)

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

fig, axes = plt.subplots(1, 2)
axes[0].imshow(ambient_rgb - res_rgb)
axes[1].imshow(flash_rgb - res_rgb)
axes[0].set_title("ambient artifacts")
axes[1].set_title("flash artifacts")
axes[0].axis("off")
axes[1].axis("off")
plt.tight_layout()
plt.show()


