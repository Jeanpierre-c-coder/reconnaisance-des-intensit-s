# -*- coding: utf-8 -*-
"""
Offline training with elementary expressions then testing with mixed expressions
from combined datasets of several participants
Rewrite By Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from sys import argv
from time import time
from random import shuffle
# - Third-party
from numpy import save
import cv2 as cv
# - Local
import utils.recog as rec
import utils.improc as imut
import utils.info as inut
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SUBJECT_IDS = ["d" + str(i) for i in range(9)]
# Replay parameters
KP_MODE = "dog"
SAW_MAX = 700


# # METHODS


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    t0 = time()
    i_iter = argv[1] if len(argv) > 1 else None
    # print(inut.make_info_dic(EXP_MODE, KP_MODE))
    # Get paths to previously recorded images and corresponding data
    train_set, _ = inut.get_img_set_multi("data/seco", SUBJECT_IDS, 1., 36, 20)
    test_set, _ = inut.get_img_set_multi("data/mix", SUBJECT_IDS, 1., 10, 40)
    # test_set, _ = inut.get_img_set_multi("data/prim", SUBJECT_IDS, 1., 25, 15)
    shuffle(train_set)
    print(len(train_set), len(test_set), i_iter)

    # FER model
    model = rec.ModelThreeway(SAW_MAX)
    # Keypoint extraction method and tools
    kp_tools, get_kp = imut.get_kp_method(KP_MODE, rec.N_KEYPOINTS)
    # Container for prediction results
    preds, truths = [], []

    # # LOOP
    for subset, is_learning in zip((train_set, test_set), (True, False)):
        for cell in subset:
            # Load and process saved image
            frame = cv.imread(cell[0], cv.IMREAD_ANYCOLOR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            kp_coord, kp_data = get_kp(gray, *kp_tools)
            if not kp_coord:
                print("No keypoints found!")

            # Feed data to model
            curr_state = cell[2] if is_learning else None
            model.clear_stm()
            img_kp = frame.copy()
            for kp_c, kp_vec in zip(kp_coord, kp_data):
                if is_learning:
                    model.feed_select(kp_vec, curr_state, cell[3])
                    model.update()
                else:
                    model.feed(kp_vec, curr_state)

            if not is_learning:
                pred_v, _ = model.predict()
                pred_l = rec.reshape_as_levels(pred_v)
                preds.append(pred_l)
                truths.append(cell[2])

    print(time() - t0)

    mats = plut.compare_by_group(preds, truths, False)

    if i_iter is not None:
        save("data/npy/mix_{}_mat.npy".format(i_iter), mats)

    print([len(saw.neurons) for saw, _ in model.ways])

    print(*mats, sep="\n")
