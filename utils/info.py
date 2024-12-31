# -*- coding: utf-8 -*-
"""
Toolbox for saving and loading sets of parameters describing experimental
conditions
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from os import makedirs, listdir
from os.path import join, isdir, isfile
from time import sleep, strftime
import json
# - Third-party
# - Local
from . import recog as rec
from . import improc as imut


# # CONSTANTS
JSON_FILENAME = "info_dic.json"
PRIMARY_DIC = {emo: rec.PRIMARY_STATES.index(emo) for emo in rec.PRIMARY_STATES}
PRIMARY_DIC["na"] = -1


# # METHODS
def make_info_dic(exp_mode, kp_mode):
    """
    Aggregate all relevant parameters into a single object describing
    experimental conditions: "info dictionary"
    """
    res = {"exp_mode": exp_mode,
           "kp_mode": kp_mode,
           "n_keypoints": rec.N_KEYPOINTS,
           "n_features": rec.N_FEATURES,
           "saw_max_size": rec.SAW_MAX_SIZE,
           "vigilance_th": rec.VIGILANCE_TH,
           "saw_lr": rec.SAW_LEARNING_RATE,
           "lms_lr": rec.LMS_LEARNING_RATE,
           "lms_minmax": (rec.LMS_MIN, rec.LMS_MAX),
           "stm_pred_ab": (rec.STM_PRED_ALPHA, rec.STM_PRED_BETA),
           "dog_halfside": imut.DOG_HALFSIDE,
           "dog_s1s2": (imut.DOG_S1, imut.DOG_S2),
           "grad_ao": (imut.GRAD_ALPHA, imut.GRAD_OMEGA),
           "polar_radius": imut.POLAR_RADIUS,
           "polar_size": imut.POLAR_SIZE,
           "filt_radius": imut.FILT_RADIUS,
           "maxima_radius": imut.MAXIMA_RADIUS}

    return res


def write_info_file(folder, info_dic):
    """
    Write the contents of an info dictionary in a human-readable JSON file
    """
    with open(join(folder, JSON_FILENAME), 'w') as ofile:
        json.dump(info_dic, ofile, indent="  ")


def read_info_file(folder):
    """
    Read a JSON file containing the contents of an info dictionary
    """
    with open(join(folder, JSON_FILENAME), 'r') as ifile:
        res = json.load(ifile)

    return res


def make_subj_folder(data_folder, subj_id):
    """
    Format unique, timestamped subject folder name and create corresponding
    directory
    """
    timestamp = strftime("%Y-%m-%d_%Hh%M")
    subj_folder = join(data_folder, "s{}_{}".format(subj_id, timestamp))
    makedirs(subj_folder)

    return subj_folder


def get_image_sets(folder, max_trials=None, max_img=None):
    """
    Get a set of images (as a list of paths) located in subject folder

    max_trials: maximum amount of trials to extract from subject folder
    max_img: maximum amount of images to extract from each trial
    """
    max_trials = max_trials if max_trials is not None else 1000
    max_img = max_img if max_img is not None else 1000  # Arbitrarily high value

    okfiles = [path for path in listdir(folder) if
               path.endswith(".png") and
               isfile(join(folder, path))]

    if len(okfiles) > 0:
        train_dic = dict()
        test_dic = dict()
        for path in okfiles:
            info = path[:-4].split("_")
            if info[2] == "unhappy":
                info[2] = "sad"

            i_trial = int(info[1])                         # Trial count
            # Trial from learning phase
            if 0 <= i_trial < max_trials:
                i_batch = int(info[2])                     # Batch count
                if i_batch < max_img:
                    val = [join(folder, path),                 # Path to image
                           PRIMARY_DIC[info[3]],               # Emotional state
                           get_levels_from_str(info[4])]       # Activation levels
                    if len(info) > 5:
                        val.append(int(info[5]))               # Learning group
                    if i_trial not in train_dic.keys():
                        train_dic[i_trial] = dict()
                    train_dic[i_trial][i_batch] = val
            else:
                test_dic[- i_trial] = join(folder, path)
    else:
        raise ValueError("No suitable image found in folder")

    # Reformat sets from dict (of dicts) to list
    train = []
    for i_t, dic in sorted(train_dic.items()):
        for i_b, val in sorted(dic.items()):
            train.append(val)
    test = []
    for i_t, path in sorted(test_dic.items()):
        test.append([path])
    # Regardless of how many learning trials were done and how many frames per
    # batch were saved, this should sort all the frames in chronological order

    return train, test


def get_img_set_multi(data_folder, subj_ids, split=None,
                      max_trials=None, max_img=None):
    """
    Get list of image sets from multiple subject folders
    """
    train_all, test_all = [], []
    for sid in subj_ids:
        subj_folder = get_subj_folder(data_folder, sid)
        train, _ = get_image_sets(subj_folder, max_trials, max_img)
        if split is None:
            train_all.extend(train)
        else:
            split_index = int(len(train) * split)
            train_all.extend(train[:split_index])
            test_all.extend(train[split_index:])

    res = train_all if split is None else train_all, test_all

    return res


def get_levels_from_str(lev_str):
    """
    Get activation levels for trait groups (as a list of lists of floats)
    corresponding to activation levels expressed by given string
    """
    res = [[0.] * n_lev for n_lev in rec.N_TRAIT_LEVELS]
    for lev, charac in zip(res, lev_str):
        act = int(charac)
        if act > 2:
            act = 1
        lev[act] = 1.

    return res


def get_subj_folder(src, subj_id):
    """
    Find and return path to subject folder inside data folder
    """
    okdirs = [path for path in listdir(src) if
              path.split("_")[0][1:] == subj_id and
              isdir(join(src, path))]
    if len(okdirs) == 1:
        res = join(src, okdirs[0])
    elif len(okdirs) > 1:
        raise ValueError("Several folders with ID: {}".format(subj_id))
    else:
        raise ValueError("No folder with ID: {}".format(subj_id))

    return res


def make_countdown(duration):
    """
    Print decreasing integers from N to 1 at the rate of one number per second
    """
    line = ""
    for sec in range(duration, 0, -1):
        line += "{}... ".format(sec)
        print(line, end="\r")
        sleep(1)
    print("\n")
