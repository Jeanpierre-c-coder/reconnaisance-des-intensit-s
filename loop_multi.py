# -*- coding: utf-8 -*-
"""
Repeat offline testing with elementary expressions from multiple participants
By Sébastien Mick rewrite by Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from subprocess import call
# - Third-party
import numpy as np
# - Local


# # CONSTANTS
# Loop parameters
N_ITER = 10
# Subprocess parameters
BASE_PARAMS = ["python3", "train_multi_sec.py"]
# Whether to save mats
SAVE_MATS = True


# # METHODS


# # MAIN
if __name__ == "__main__":
    for ind in range(N_ITER):
        params = BASE_PARAMS + [str(ind)] if SAVE_MATS else BASE_PARAMS
        call(params)

    if SAVE_MATS:
        multi_mats = []
        for ind in range(N_ITER):
            multi_mats.append(np.load("data/npy/multi_{}_mat.npy".format(ind)))
        mat_arr = np.array(multi_mats)
        # Shape n_rerolls * n_groups * n_levels * n_levels
        print(mat_arr.shape)
        np.save("data/npy/r{}_multi8_mat.npy".format(N_ITER), mat_arr)
