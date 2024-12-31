# -*- coding: utf-8 -*-
"""
Toolbox for plotting various relevant figures
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
# - Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as pcm
# - Local
import utils.recog as rec


# # CONSTANTS
MODE_COLORS = ["r", "g", "b"]


# # METHODS
def plot_conf_mat(mat, block=False):
    """
    Plot 2D graphs illustrating recognition rates, given in percents over the
    whole matrix
    """
    fig, (ax_bar, ax_mat) = plt.subplots(1, 2)

    # Bar plot for recognition rates by state
    rates_by_state = np.diag(mat) / np.sum(mat, axis=1)
    xticks = list(range(mat.shape[0]))
    ax_bar.bar(xticks, rates_by_state)
    for xpos, rate in enumerate(rates_by_state):
        ax_bar.text(x=xpos, y=rate + 0.04, s="{:.2}".format(rate),
                    va="center", ha="center", size="medium")
    ax_bar.set_xticks(xticks)
    if len(rates_by_state) == len(rec.PRIMARY_STATES):
        ax_bar.set_xticklabels(rec.PRIMARY_STATES)
    ax_bar.set_xlabel("Affective state")
    ax_bar.set_ylabel("Recognition rate")
    ax_bar.set_ylim(0, 1)

    # Confusion matrix
    ax_mat.matshow(mat, cmap=pcm.get_cmap("Blues"), alpha=0.7)
    fmat = np.round(100 * mat / np.sum(mat), 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_mat.text(x=j, y=i, s=fmat[i, j],
                        va="center", ha="center", size="large")

    ax_mat.set_xlabel("Predictions")
    ax_mat.set_ylabel("Ground truth")

    plt.show(block=block)


def plot_conf_mat2(mat, block=False):
    """
    Plot 2D graphs illustrating recognition rates, given in percents over the
    current line
    """
    fig, (ax_bar, ax_mat) = plt.subplots(1, 2)

    # Bar plot for recognition rates by state
    freq_by_state = mat / np.expand_dims(np.sum(mat, axis=1), 1)
    ok_rates = np.diag(freq_by_state)
    xticks = list(range(mat.shape[0]))
    ax_bar.bar(xticks, ok_rates)
    for xpos, rate in enumerate(ok_rates):
        ax_bar.text(x=xpos, y=rate + 0.04, s="{:.2}".format(rate),
                    va="center", ha="center", size="medium")
    ax_bar.set_xticks(xticks)
    if len(ok_rates) == len(rec.PRIMARY_STATES):
        ax_bar.set_xticklabels(rec.PRIMARY_STATES)
    ax_bar.set_xlabel("Affective state")
    ax_bar.set_ylabel("Recognition rate")
    ax_bar.set_ylim(0, 1)

    # Confusion matrix
    ax_mat.matshow(freq_by_state, cmap=pcm.get_cmap("Blues"), alpha=0.7)
    fmat = np.round(100 * freq_by_state, 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_mat.text(x=j, y=i, s=fmat[i, j],
                        va="center", ha="center", size="large")

    ax_mat.set_xlabel("Predictions")
    ax_mat.set_ylabel("Ground truth")

    plt.show(block=block)


def plot_growth(vals, ax=None, block=False):
    """
    Plot a 2D graph illustrating neuron growth in SAW layer over trials
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(vals)
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Number of SAW neurons")

    plt.show(block=block)


def plot_hist(hist, block=False):
    """
    Plot histogram of activations on SAW layer
    """
    ymax = max(hist)
    ymed = np.median(hist)
    quartiles = np.array([0.25, 0.5, 0.75]) * sum(hist)
    cumul = np.cumsum(hist)

    _, ax = plt.subplots(1, 1)
    ax.plot(hist, "bx")
    ax.plot([0, len(hist)], [ymed, ymed])
    for val in quartiles:
        bool_arr = cumul >= val
        xpos = np.argmax(~bool_arr[:-1] & bool_arr[1:]) + 1
        ax.plot([xpos, xpos], [0, ymax])

    plt.show(block=block)


def plot_saw(model, hist=False, block=False):
    """
    Plot relevant graphs for single or multiple SAW layers of given model
    """
    _, gax = plt.subplots(1, 1)

    try:
        layers = [model.saw]
    except AttributeError:
        layers = [saw for saw, _ in model.ways]

    for ind, lay in reversed(list(enumerate(layers))):
        if hist:
            plot_hist(lay.hist, False)

        plot_growth(lay.growth, gax, ind == 0 and block)


def plot_recog_rate_box(rec_arr, block=False):
    """
    Boxplot illustrating distribution of recognition rates over groups and
    activation levels
    """
    # One axis for each group
    _, axes = plt.subplots(1, rec_arr.shape[1], sharex="all", sharey="all")
    for ind, ax in enumerate(axes):
        ax.boxplot(rec_arr[:, ind, :], whis=[0, 100], positions=[-1, 0, 1])
        ax.set_ylim(-0.05, 1.05)

    plt.show(block=block)

    return axes


def plot_recog_errorbar(rec_arr, axes=None, block=False):
    """
    Errorbar graph illustrating mean and standard deviation of recognition rates
    over groups and activation levels
    """
    if axes is None:
        _, axes = plt.subplots(1, rec_arr.shape[1], sharex="all", sharey="all")

    xpos = [-0.75, 0.25, 1.25]
    for ind, ax in enumerate(axes):
        means = np.mean(rec_arr[:, ind, :], axis=0)
        sds = np.std(rec_arr[:, ind, :], axis=0)
        ax.errorbar(xpos, means, yerr=sds, fmt="bo", capsize=4,
                    markersize=5, markerfacecolor="b")
        ax.set_ylim(-0.05, 1.15)

    plt.show(block=block)


def plot_recog_compare(rec_arr, block=False):
    """
    Boxplot graph of recognition rates over groups and modes
    """
    n_modes, n_groups, n_levels = rec_arr.shape[1:4]
    xpos_ref = np.arange(n_levels) * (n_modes + 1)
    for i_grp in range(n_groups):
        _, ax = plt.subplots(1, 1)
        for i_mod, col in enumerate(MODE_COLORS):
            xpos = xpos_ref + i_mod
            ax.boxplot(rec_arr[:, i_mod, i_grp, :], positions=xpos,
                       boxprops={"color": col}, whiskerprops={"color": col},
                       capprops={"color": col}, whis=[0, 100])
        for xline in xpos_ref[:-1] + 3:
            ax.plot((xline, xline), (0, 1), "k-")
        ax.set_xticks([xt + 1 for xt in xpos_ref])
        ax.set_xticklabels(rec.TRAIT_LEVELS)
        ax.set_ylim(0., 1.)

    plt.show(block=block)


def compare_by_group(pred_list, truth_list, show=False):
    """
    Compare predicted activation levels to ground truth by dissociating trait
    groups
    """
    n_preds = len(pred_list)
    assert(n_preds == len(truth_list))

    # Single-pass formatting into arrays (one per group) of pred/truth pairs
    pairs_lists = [[] for _ in rec.TRAIT_GROUPS]
    for p_lev, t_lev in zip(pred_list, truth_list):
        for i_group in range(rec.N_GROUPS):
            pairs_lists[i_group].append((p_lev[i_group], t_lev[i_group]))
    # Array is shape n_groups * n_preds * 2
    argm_arr = np.array([np.argmax(pairs_list, axis=2)
                         for pairs_list in pairs_lists])
    sum_arr = np.zeros(n_preds, dtype=int)

    mats = []
    for arr, n_lev in zip(argm_arr, rec.N_TRAIT_LEVELS):
        conf_mat = np.zeros((n_lev, n_lev), dtype=int)
        # Raw recognition rate for this traits group
        diff = arr[:, 0] == arr[:, 1]
        sum_arr += diff
        # Confusion matrix
        for pred, truth in arr:
            conf_mat[truth, pred] += 1
        if show:
            print("{:.2}".format(np.mean(diff)))
            print(conf_mat)
        mats.append(conf_mat)

    return mats


def compare_by_likely_state(pred_list, truth_list):
    """
    Compare predicted activation levels to ground truth
    """
    assert(len(pred_list) == len(truth_list))

    pairs = [(rec.get_likely_emo_state(pred), rec.get_likely_emo_state(truth))
             for pred, truth in zip(pred_list, truth_list)]
    diff = [pred_s == truth_s for pred_s, truth_s in pairs]
    print(np.mean(diff))
    conf_mat = np.zeros((rec.N_STATES, rec.N_STATES), dtype=int)
    for pred, truth in pairs:
        conf_mat[truth, pred] += 1
    print(conf_mat)

    return pairs


def get_nn_rates(mats):
    """
    From confusion matrices, compute recognition rates of non-neutral activation
    levels only
    """
    res = []
    for mat, n_lev in zip(mats, rec.N_TRAIT_LEVELS):
        rate = 0
        for i_row in set(range(n_lev)) - {1}:
            row = mat[i_row, :]
            rate += row[i_row] / sum(row)
        res.append(np.round(rate / (n_lev - 1), 3))

    return res
