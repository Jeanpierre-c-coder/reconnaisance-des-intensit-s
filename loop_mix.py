# -*- coding: utf-8 -*-
"""
Repeat offline testing with mixed expressions
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
N_ITER = 20
# Subprocess parameters
BASE_PARAMS = ["python3", "train_multi_mix.py"]
# Whether to save mats
SAVE_MATS = True


# # METHODS


# # MAIN
if __name__ == "__main__":
    for ind in range(N_ITER):
        params = BASE_PARAMS + [str(ind)] if SAVE_MATS else BASE_PARAMS
        call(params)

    if SAVE_MATS:
        mix_mats = []
        for ind in range(N_ITER):
            mix_mats.append(np.load("data/npy/mix_{}_mat.npy".format(ind)))
        mat_arr = np.array(mix_mats)
        # Shape n_rerolls * n_groups * n_levels * n_levels
        print(mat_arr.shape)
        np.save("data/npy/r{}_mix_mat.npy".format(N_ITER), mat_arr)
