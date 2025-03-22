# -*- coding: utf-8 -*-
"""
Process saved confusion matrices to generate plots suitable for publication
By Sébastien Mick rewrite by Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from itertools import combinations
# - Third-party
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
import matplotlib.cm as pcm
# - Local


# # CONSTANTS
TRAIT_LEVELS = ["Low", "Neutral", "High"]
TRAIT_GROUPS = ["Eyebrows", "Lip corners", "Mouth opening"]
MODES = ["OR", "TH", "BH"]
XPOS = [0, 1, 2]
CHANCE = 1. / 3.
DO_STAT = True


# # METHODS
def conf_mat(mat):
    freq_by_state = mat / np.expand_dims(np.sum(mat, axis=1), 1)
    freq_round = np.round(freq_by_state, 2)
    _, axm = plt.subplots(1, 1)
    axm.matshow(freq_by_state, cmap=pcm.get_cmap("Blues"), alpha=0.7,
                vmin=0., vmax=1.)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axm.text(x=j, y=i, s=freq_round[i, j],
                     va="center", ha="center", size="large")

    axm.set_xlabel("Predictions")
    axm.set_ylabel("Ground truth")
    axm.set_xticks(XPOS)
    axm.set_yticks(XPOS)
    axm.set_xticklabels(TRAIT_LEVELS)
    axm.set_yticklabels(TRAIT_LEVELS)


def success_bars(mat, axb=None, colors=None, xlabels=None,
                 leg=False, ylab=False):
    cols = ("darkcyan", "indianred", "seagreen") if colors is None else colors

    diag_count = np.sum(mat[:, :, diag_ind, diag_ind], axis=2)
    rec_rate = diag_count / np.sum(mat, axis=(2, 3))
    means = 100 * np.mean(rec_rate, axis=0)
    sds = 100 * np.std(rec_rate, axis=0)
    print(np.round(means, 4), np.round(sds, 4))

    if axb is None:
        _, axb = plt.subplots(1, 1)
    axb.bar(XPOS, means, width=0.8, color=cols)
    axb.errorbar(XPOS, means, yerr=sds, fmt="ko", markerfacecolor="k",
                 capsize=12, markersize=5)
    axb.plot([-5, 5], [100. / 3.] * 2, "k--", label="Chance level")
    axb.set_ylim(0., 100.)
    axb.set_xlim(-0.6, 2.6)
    axb.set_xticks(XPOS)
    if xlabels is not None:
        axb.set_xticklabels(xlabels)
    if leg:
        axb.legend()
    if ylab:
        axb.set_ylabel("Success rate [%]")

    return rec_rate


# # MAIN
if __name__ == "__main__":
    all_arr = np.load("data/npy/mask_r20_TB_mat_d.npy")
    # Shape n_rerolls * n_modes * n_groups * n_levels * n_levels
    print("Mask", all_arr.shape)
    n_groups, n_levels = all_arr.shape[2:4]
    diag_ind = np.arange(0, n_levels)

    # # MULTI
    # mat_arr = all_arr[:, 0, :, :, :]
    # success_bars(mat_arr, None, TRAIT_GROUPS)
    # Loop over groups to plot confusion matrices
    # for mat_sum in np.sum(mat_arr, axis=0):
    #    conf_mat(mat_sum)

    # # MASK
    rates_or = []
    _, axeb = plt.subplots(1, 1)
    _, axmouth = plt.subplots(1, 2, sharey=True)
    params = ((axeb, True, True),
              (axmouth[0], False, True),
              (axmouth[1], True, False))

    for (ax, lege, yl), (i_grp, lab) in zip(params, enumerate(TRAIT_GROUPS)):
        mat_arr = all_arr[:, :, i_grp, :, :]
        rates = success_bars(mat_arr, ax, ("tab:red", "tab:blue", "tab:green"),
                             MODES, lege, yl)
        ax.set_title(lab)

        # Loop over modes
        # for mat_sum in np.sum(mat_arr, axis=0):
        #    conf_mat(mat_sum)

        rates_or.append(rates[:, 0])

        if DO_STAT:
            for i_mod in range(3):
                n, pn = stat.normaltest(rates[:, i_mod])
                s, ps = stat.ttest_1samp(rates[:, i_mod], CHANCE,
                                         alternative="greater")
                print(ps)
            a, pa = stat.f_oneway(*rates.T)
            # k, pk = stat.kruskal(*rates.T)
            print(lab, pa)
            for i_mod, j_mod in combinations(range(3), 2):
                # w, pw = stat.wilcoxon(rates[:, i_mod], rates[:, j_mod])
                t, pt = stat.ttest_rel(rates[:, i_mod], rates[:, j_mod])
                print(i_mod, j_mod, pt)

    # # MIX
    mix_arr = np.load("data/npy/r20_mix_mat.npy")
    # Shape n_rerolls * n_groups * n_levels * n_levels
    print("Mix", mix_arr.shape)
    _, axmix = plt.subplots(1, 1)
    rates_mix = success_bars(mix_arr, axmix, None, TRAIT_GROUPS, True, True)

    if DO_STAT:
        for rate_grp, rate_or in zip(rates_mix.T, rates_or):
            n, pn = stat.normaltest(rate_grp)
            s, ps = stat.ttest_1samp(rate_grp, CHANCE, alternative="greater")
            print(pn, ps)
            t, pt = stat.ttest_ind(rate_grp, rate_or)
            print(pt)

    plt.show(block=True)
