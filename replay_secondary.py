# -*- coding: utf-8 -*-
"""
Offline training/testing with elementary expressions from the dataset of a
single participant
By Sébastien Mick
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
import matplotlib.pyplot as plt
import utils.info as inut
import utils.recog as rec
import utils.improc as imut
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SRC_FOLDER = "data/seco"
SUBJECT_ID = "c0"
# Replay parameters
EXP_MODE = 3
KP_MODE = "dog"
SHOW = False
# Color coding of trait groups
COLOR_CODES = ((255, 0, 0),    # Blue = eyebrows
               (0, 255, 0),    # Green = lip corners
               (0, 0, 255),    # Red = mouth opening
               (0, 255, 255))  # Yellow = none


# # METHODS


# # MAIN
if __name__ == "__main__":
    t0 = time()
    # # PREPARE
    save_mats = len(argv) > 2
    sid = argv[1] if len(argv) > 1 else SUBJECT_ID
    subj_folder = inut.get_subj_folder(SRC_FOLDER, sid)

    # Get paths to previously recorded images and corresponding data
    annot_set, _ = inut.get_image_sets(subj_folder, 36, 15)
    split = len(annot_set) * 3 // 4
    img_sets = [annot_set[:split], annot_set[split:]]
    shuffle(img_sets[0])
    print(len(img_sets[0]))

    # FER model
    model = rec.ModelThreeway() if EXP_MODE == 3 else rec.ModelByGroup()
    # Keypoint extraction method and tools
    kp_tools, get_kp = imut.get_kp_method(KP_MODE, rec.N_KEYPOINTS)
    # Containers for results
    preds, truths = [], []
    learning_finished = False

    # # LOOPS
    for img_set, is_learning in zip(img_sets, (True, False)):
        for ind, cell in enumerate(img_set):
            # Load and process saved image
            frame = cv.imread(cell[0], cv.IMREAD_ANYCOLOR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Hide (>) or show (<=) half corresponding to trait group
            mask = None #  if is_learning or EXP_MODE != 3 else int(cell[3] <= 0)
            # Hide top (0) or bottom (1) half
            # mask = None if is_learning or EXP_MODE != 3 else 1
            kp_coord, kp_data = get_kp(gray, *kp_tools, mask)
            if not kp_coord:
                print("No keypoints found!")

            # Feed data to model
            levels = cell[2]
            curr_state = levels if is_learning else None
            model.clear_stm()
            img_kp = frame.copy()
            for kp_c, kp_vec in zip(kp_coord, kp_data):
                if EXP_MODE in (2, 3) and is_learning:
                    model.feed_select(kp_vec, curr_state, cell[3])
                else:
                    model.feed(kp_vec, curr_state)
                if is_learning:
                    model.update()

                elif SHOW and EXP_MODE != 3:
                    max_group = rec.get_highest_group(model.lms_layers)
                    img_kp = imut.draw_kp(img_kp, [kp_c],
                                          [COLOR_CODES[max_group]])

            if not is_learning:
                if not learning_finished:
                    learning_finished = True
                    # print(len(model.saw.neurons))
                pred_v, _ = model.predict()
                pred_l = rec.reshape_as_levels(pred_v)
                preds.append(pred_l)
                truths.append(levels)
                if SHOW:
                    if EXP_MODE == 3:
                        print(cell[3], mask)
                        img_kp = imut.draw_kp(img_kp, kp_coord, (0, 0, 255))
                    print("Truth:", rec.project_3d(levels),
                          "| Pred:", rec.project_3d(pred_l))
                    cv.imshow("Image with keypoints", img_kp)
                    char = cv.waitKey(-1) & 0xFF
                    if char == ord('q'):  # Exit program
                        raise KeyboardInterrupt
                    elif char == ord('s'):  # Skip to results
                        cv.destroyAllWindows()
                        SHOW = False

    plot = False
    mats = plut.compare_by_group(preds, truths, show=False)
    if plot:
        print(time() - t0)
        for i_grp, mat in enumerate(mats):
            plut.plot_conf_mat(mat)
            # plt.savefig("img/{}_{}.png".format(sid, rec.TRAIT_GROUPS[i_grp]),
            #            dpi=150)
        plut.plot_saw(model, block=True)
        #plt.savefig("img/{}_saw.png".format(sid), dpi=150)
    else:
        rates = plut.get_nn_rates(mats)
        print(rates)
        print([len(saw.neurons) for saw, _ in model.ways])

    if save_mats:
        save("data/npy/{}_mat.npy".format(sid), mats)
