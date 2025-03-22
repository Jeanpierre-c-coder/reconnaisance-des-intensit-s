# -*- coding: utf-8 -*-
"""
Process saved confusion matrices to investigate recognition rates obtained in
various conditions
By Sébastien Mick rewrite by Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from os import listdir
from os.path import join, isfile
# - Third-party
import numpy as np
import matplotlib.pyplot as plt
# - Local
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SUBJECT_IDS = ["c{}".format(i) for i in range(8)]
# ["b" + str(i) for i in range(10)] +


# # METHODS
def get_mat_array(src_folder):
    """
    Find and load files storing confusion matrices, then aggregate as one array
    """
    okfiles = [path for path in listdir(src_folder) if
               path.endswith("_all_mat.npy") and
               isfile(join(src_folder, path))]
    res = []
    for ifile in okfiles:
        res.append(np.load(join(src_folder, ifile)))

    return np.array(res)


# # MAIN
if __name__ == "__main__":
    # # SINGLE
    mat_arr = get_mat_array("data/npy/")
    # Shape n_rerolls * n_subjects * n_groups * n_levels * n_levels
    print("Single", mat_arr.shape)
    n_levels = mat_arr.shape[-1]

    # Compute recognition rates
    diag_ind = np.arange(0, n_levels)
    diag_arr = mat_arr[:, :, :, diag_ind, diag_ind]
    rec_arr = diag_arr / np.sum(mat_arr, axis=4)

    # Reshape over subjects and rerolls
    new_shape = [rec_arr.shape[0] * rec_arr.shape[1], *rec_arr.shape[2:]]
    resh_arr = np.reshape(rec_arr, new_shape)
    # axes = plut.plot_recog_rate_box(resh_arr, False)
    # plut.plot_recog_errorbar(resh_arr, axes, False)

    # # MULTI
    for label, fname in zip(("Multi", "LOO", "Mix"),
                            ("data/npy/r10_multi_mat.npy",
                             # "data/npy/all_loo_mat.npy",
                             "data/npy/r20_mix_mat.npy")):
        mat_arr = np.load(fname)
        # Shape n_rerolls * n_groups * n_levels * n_levels
        print(label, mat_arr.shape)

        # Compute recognition rates
        diag_arr = mat_arr[:, :, diag_ind, diag_ind]
        sum_mat = np.sum(mat_arr, axis=3)
        sum_mat[sum_mat == 0] = 1.
        rec_arr = diag_arr / sum_mat

        axes = plut.plot_recog_rate_box(rec_arr, False)
        plut.plot_recog_errorbar(rec_arr, axes, False)

    # # MASK
    mat_arr = np.load("data/npy/mask_r10_TB_mat_b.npy")
    # Shape n_rerolls * n_modes * n_groups * n_levels * n_levels
    print("Mask", mat_arr.shape)
    diag_arr = mat_arr[:, :, :, diag_ind, diag_ind]
    rec_arr = diag_arr / np.sum(mat_arr, axis=3)

    # plut.plot_recog_compare(rec_arr, True)
    plt.show(block=True)
