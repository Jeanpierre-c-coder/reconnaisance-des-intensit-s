# -*- coding: utf-8 -*-
"""
Offline training/testing with primary emotions from combined datasets of several
participants
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from time import time
from random import shuffle
# - Third-party
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# - Local
import utils.recog as rec
import utils.improc as imut
import utils.info as inut
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SUBJECT_IDS = ["b9", "c1", "c2", "c7", "b0", "b1", "b2"]
SPLIT_RATIO = 0.75
# Replay parameters
EXP_MODE = 0
KP_MODE = "dog"
SAW_MAX = 200


# # METHODS


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    t0 = time()
    print(inut.make_info_dic(EXP_MODE, KP_MODE))
    # Get paths to previously recorded images and corresponding data
    train_set, test_set = inut.get_img_set_multi("data/seco", SUBJECT_IDS,
                                                 SPLIT_RATIO, 36, 10)
    shuffle(train_set)
    print(len(train_set))

    # Keypoint extraction method and tools
    kp_tools, get_kp = imut.get_kp_method(KP_MODE, rec.N_KEYPOINTS)
    # FER model
    model = rec.ModelByState(SAW_MAX)
    # Container for prediction results
    pred_results = []

    extra_ct = [0, 0, 0]

    # # LOOP
    for subset, is_learning in zip((train_set, test_set), (True, False)):
        for cell in subset:
            if is_learning and cell[1] < 3:
                if cell[1] == 0:
                    if extra_ct[0] >= 60 * len(SUBJECT_IDS):
                        continue
                    else:
                        extra_ct[0] += 1
                if cell[1] in (1, 2) and extra_ct[cell[1]] < 30 * len(SUBJECT_IDS):
                    subset.append(cell)
                    extra_ct[cell[1]] += 1
            # Load and process saved image
            frame = cv.imread(cell[0], cv.IMREAD_ANYCOLOR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            kp_coord, kp_data = get_kp(gray, *kp_tools)
            if not kp_coord:
                print("No keypoints found!")
            # Read info from file name
            curr_state = cell[1] if is_learning else None
            # Feed data to model
            model.clear_stm()
            for kp_vec in kp_data:
                model.feed(kp_vec, curr_state)
                if is_learning:
                    model.update()

            if not is_learning:
                _, pred_emo = model.predict()
                pred_results.append((pred_emo, cell[1]))

    print(time() - t0)
    print(len(model.saw.neurons))
    print(extra_ct)

    recogs = [pred == truth for pred, truth in pred_results]
    recog_rate = int(np.mean(recogs) * 100)
    conf_matrix = np.zeros([rec.N_STATES] * 2, dtype=int)
    for pred, truth in pred_results:
        conf_matrix[truth, pred] += 1
    plut.plot_conf_mat(conf_matrix)
    plt.savefig("img/{}_prim_double.png".format(SUBJECT_IDS[0]), dpi=150)
