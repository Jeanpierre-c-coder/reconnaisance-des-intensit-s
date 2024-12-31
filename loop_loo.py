# -*- coding: utf-8 -*-
"""
Repeat offline testing with Leave-One-Out strategy
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from subprocess import call
# - Third-party
import numpy as np
# - Local


# # CONSTANTS
# Loop parameters
N_SUBJECTS = 18
# Subprocess parameters
BASE_PARAMS = ["python3", "train_loo.py"]
# Whether to save mats
SAVE_MATS = True


# # METHODS


# # MAIN
if __name__ == "__main__":
    for ind in range(N_SUBJECTS):
        params = BASE_PARAMS + [str(ind)]
        if SAVE_MATS:
            params.append("1")
        call(params)

    if SAVE_MATS:
        loo_mats = []
        for ind in range(N_SUBJECTS):
            loo_mats.append(np.load("data/npy/loo_{}_mat.npy".format(ind)))
        # Shape n_rerolls * n_groups * n_levels * n_levels
        mat_arr = np.array(loo_mats)
        print(mat_arr.shape)
        np.save("data/npy/all_loo_mat.npy", mat_arr)
