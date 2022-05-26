
import numpy as np
from skimage.segmentation import find_boundaries

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