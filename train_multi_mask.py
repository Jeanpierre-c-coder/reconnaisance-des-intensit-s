# -*- coding: utf-8 -*-
"""
Offline training/testing with elementary expressions from combined datasets of
several participants, and partial masking
Rewrite By Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from sys import argv
from time import time
from random import shuffle
# - Third-party
import numpy as np
import cv2 as cv
# - Local
import utils.recog as rec
import utils.improc as imut
import utils.info as inut
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SUBJECT_IDS = ["c" + str(i) for i in range(8)] + ["b" + str(i) for i in range(10)] + ["d" + str(i) for i in range(9)]
SPLIT_RATIO = 0.75
# Replay parameters
EXP_MODE = 3
KP_MODE = "dog"
SAW_MAX = 1900
# Mask parameters
MASK_DIC = {"L": [None] * 3,  # Learning: mask nothing
            "O": [None] * 3,  # Original
            "T": [0] * 3,     # Top half
            "B": [1] * 3,     # Bottom half
            "RH": [0, 1, 1],  # Relevant half (top for EB, bottom for LC and MO)
            "IH": [1, 0, 0]}  # Irrelevant half


# # METHODS


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    t0 = time()
    i_iter = argv[1]
    mask_modes = argv[2:]
    # Get paths to previously recorded images and corresponding data
    train_set, test_set = inut.get_img_set_multi("data/seco", SUBJECT_IDS,
                                                 SPLIT_RATIO, 36, 10)
    shuffle(train_set)
    print(len(train_set), i_iter, mask_modes)

    # FER model
    model = rec.ModelThreeway(SAW_MAX)
    # Keypoint extraction method and tools
    kp_tools, get_kp = imut.get_kp_method(KP_MODE, rec.N_KEYPOINTS)
    # Loop parameters
    modes = ["L", "O"] + mask_modes
    # Container for prediction results
    pred_container = [[[], []] for _ in modes]

    # # LOOP
    for mode, (preds, truths) in zip(modes, pred_container):
        subset = train_set if mode == "L" else test_set
        for cell in subset:
            # Load and process saved image
            frame = cv.imread(cell[0], cv.IMREAD_ANYCOLOR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            kp_coord, kp_data = get_kp(gray, *kp_tools,
                                       MASK_DIC[mode][cell[3]])
            if not kp_coord:
                print("No keypoints found!")

            # Feed data to model
            curr_state = cell[2] if mode == "L" else None
            model.clear_stm()
            img_kp = frame.copy()
            for kp_vec in kp_data:
                if mode == "L":
                    model.feed_select(kp_vec, curr_state, cell[3])
                    model.update()
                else:
                    model.feed(kp_vec, curr_state)

            if not mode == "L":
                pred_v, _ = model.predict()
                pred_l = rec.reshape_as_levels(pred_v)
                preds.append(pred_l)
                truths.append(cell[2])

    print(time() - t0, [len(saw.neurons) for saw, _ in model.ways])
    mat_list = []
    for preds, truths in pred_container[1:]:
        mat_list.append(plut.compare_by_group(preds, truths, False))
    mat_arr = np.array(mat_list)

    np.save("data/npy/mask_{}_{}_mat.npy".format(i_iter, "".join(mask_modes)),
            mat_arr)
