# -*- coding: utf-8 -*-
"""
Offline training/testing with elementary expressions from combined datasets of
several participants, following a Leave-One-Out strategy
Rewrite By Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from sys import argv
from time import time
from random import shuffle
# - Third-party
from numpy import save
# import cv2 as cv
import matplotlib.pyplot as plt
# - Local
import utils.recog as rec
import utils.improc as imut
import utils.info as inut
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SUBJECT_IDS = ["b" + str(i) for i in range(10)] + ["c{}".format(i) for i in range(8)]
TEST_ID_IND = 1
# Replay parameters
EXP_MODE = 3
KP_MODE = "dog"
SAW_MAX = 1800


# # METHODS


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    t0 = time()
    save_mats = len(argv) > 2
    iid = int(argv[1]) if len(argv) > 1 else TEST_ID_IND
    sid = SUBJECT_IDS[iid]
    # print(inut.make_info_dic(EXP_MODE, KP_MODE))
    # Get paths to previously recorded images and corresponding data
    SUBJECT_IDS.remove(sid)
    test_set = inut.get_image_sets(inut.get_subj_folder("data/seco", sid))[0]
    train_set = inut.get_img_set_multi("data/seco", SUBJECT_IDS, 1., 36, 10)[0]
    print(len(train_set), len(test_set), sid)
    shuffle(train_set)

    # FER model
    model = rec.ModelThreeway(SAW_MAX) if EXP_MODE == 3 else rec.ModelByGroup(SAW_MAX)
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
                if EXP_MODE in (2, 3) and is_learning:
                    model.feed_select(kp_vec, curr_state, cell[3])
                else:
                    model.feed(kp_vec, curr_state)
                if is_learning:
                    model.update()

            if not is_learning:
                pred_v, _ = model.predict()
                pred_l = rec.reshape_as_levels(pred_v)
                preds.append(pred_l)
                truths.append(cell[2])

    print(time() - t0)

    mats = plut.compare_by_group(preds, truths, False)
#    for i_grp, mat in enumerate(mats):
#        plut.plot_conf_mat(mat)
#        plt.savefig("img/loo_{}.png".format(rec.TRAIT_GROUPS[i_grp]), dpi=150)
#    plut.plot_saw(model, block=True)
#    plt.savefig("img/loo_saw.png", dpi=150)

    if save_mats:
        save("data/npy/loo_{}_mat.npy".format(iid), mats)

    print([len(saw.neurons) for saw, _ in model.ways])
