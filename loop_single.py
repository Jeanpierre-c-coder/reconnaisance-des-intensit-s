# -*- coding: utf-8 -*-
"""
Repeat offline testing on a single participant's dataset
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from subprocess import call
# - Third-party
import numpy as np
# - Local
import utils.info as inut


# # CONSTANTS
# Loading parameters
SUBJECT_IDS = ["b" + str(i) for i in range(10)] + ["c{}".format(i) for i in range(8)]
# Subprocess parameters
BASE_PARAMS = ["python3", "replay_secondary.py"]
# Whether to save mats
SAVE_MATS = False


# # METHODS


# # MAIN
if __name__ == "__main__":
    print(inut.make_info_dic(-1, -1))
    for sid in SUBJECT_IDS:
        print("\n", sid)
        params = BASE_PARAMS + [sid]
        if SAVE_MATS:
            params.append("1")
        call(params)

    if SAVE_MATS:
        all_mats = []
        for sid in SUBJECT_IDS:
            all_mats.append(np.load("data/npy/{}_mat.npy".format(sid)))
        # Shape n_subjects * n_groups * n_levels * n_levels
        mat_arr = np.array(all_mats)
        np.save("data/npy/9_all_mat.npy", mat_arr)
