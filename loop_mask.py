# -*- coding: utf-8 -*-
"""
Repeat offline testing with given partial masking modalities
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from subprocess import call
# - Third-party
import numpy as np
# - Local


# # CONSTANTS
# Subprocess parameters
N_REPEATS = 20
BASE_PARAMS = ["python3", "train_multi_mask.py"]
MASK_MODES = ["T", "B"]
MASK_STR = "".join(MASK_MODES)


# # METHODS


# # MAIN
if __name__ == "__main__":
    for ind in range(N_REPEATS):
        params = BASE_PARAMS + [str(ind)]
        call(params + MASK_MODES)

    all_mats = []
    for ind in range(N_REPEATS):
        pair = np.load("data/npy/mask_{}_{}_mat.npy".format(ind, MASK_STR))
        all_mats.append(pair)
    # Shape n_rerolls * n_modes * n_groups * n_levels * n_levels
    mat_arr = np.array(all_mats)
    np.save("data/npy/mask_r{}_{}_mat_d.npy".format(N_REPEATS, MASK_STR), mat_arr)
