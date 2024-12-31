# -*- coding: utf-8 -*-
"""
Pre-process recorded images and save outputs from visual processing
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from pickle import dump, load
# - Third-party
import cv2 as cv
# - Local
from .improc import get_kp_method
from . import info as inut


# # CONSTANTS
BID = ["b" + str(i) for i in range(10)]
CID = ["c" + str(i) for i in range(8)]
DID = ["d" + str(i) for i in range(9)]
MODES = {"prim": DID,
         "seco": BID + CID + DID,
         "mix": DID}
dic_file = "data/all_dic.pkl"


# # METHODS
def load_all_dic():
    """
    Load pickle file storing preprocessed descriptors
    """
    with open(dic_file, "rb") as ifile:
        res = load(ifile)

    return res


def get_set_multi(mode_dic, subj_ids,
                  split=None, max_trials=None, max_img=None):
    """
    Load and split datasets from multiple subjects
    """
    train_all, test_all = [], []

    for sid in subj_ids:
        subj_dic = mode_dic[sid]
        train_dic = dict()
        for path, des in subj_dic.items():
            info = path[:-4].split("_")

            i_trial = int(info[1])                             # Trial count
            if 0 <= i_trial < max_trials:
                i_batch = int(info[2])                         # Batch count
                if i_batch < max_img:
                    val = [des,                                # Descriptor
                           inut.PRIMARY_DIC[info[3]],          # Emotional state
                           inut.get_levels_from_str(info[4])]  # Levels
                    if len(info) > 5:
                        val.append(int(info[5]))               # Learning group
                    if i_trial not in train_dic.keys():
                        train_dic[i_trial] = dict()
                    train_dic[i_trial][i_batch] = val
        train = []
        for i_t, dic in sorted(train_dic.items()):
            for i_b, val in sorted(dic.items()):
                train.append(val)

        if split is None:
            train_all.extend(train)
        else:
            split_index = int(len(train) * split)
            train_all.extend(train[:split_index])
            test_all.extend(train[split_index:])

    res = train_all if split is None else train_all, test_all

    return res


# # MAIN
if __name__ == '__main__':
    kp_tools, get_kp = get_kp_method("dog", 15)
    # 3 modes * 9-27 subjects * 240-1080 images * (15 kp * 81 px)
    all_dic = dict()
    for mode, subs in MODES.items():
        mod_dic = dict()
        for sub_id in subs:
            try:
                sub_dir = inut.get_subj_folder("data/{}/".format(mode), sub_id)
            except ValueError:
                continue
            sub_dic = dict()
            lab_set, _ = inut.get_image_sets(sub_dir)
            for cell in lab_set:
                frame = cv.imread(cell[0], cv.IMREAD_ANYCOLOR)
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                _, kp_data = get_kp(gray, *kp_tools)
                filename = cell[0].split('/')[-1]
                sub_dic[filename] = kp_data.tolist()

            print(sub_id, len(sub_dic))
            mod_dic[sub_id] = sub_dic

        print(mode, len(mod_dic))
        all_dic[mode] = mod_dic

    with open(dic_file, "wb") as ofile:
        dump(all_dic, ofile)
