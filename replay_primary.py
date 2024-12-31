# -*- coding: utf-8 -*-
"""
Offline training/testing with primary emotions from the dataset of a single
participant
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from sys import argv
from os.path import join
from random import shuffle
# - Third-party
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# - Local
import utils.info as inut
import utils.recog as rec
import utils.improc as imut
import utils.plot as plut


# # CONSTANTS
# Loading parameters
SRC_FOLDER = "data/prim"
SUBJECT_ID = "b0"
# Replay parameters
KP_MODE = "dog"
SHOW = False
# Color coding of expressions
COLOR_CODES = ((255, 0, 0),    # Blue = neutral
               (0, 255, 0),    # Green = happy
               (0, 0, 255),    # Red = sad
               (255, 0, 255),  # Purple = surprised
               (0, 255, 255))  # Yellow = angry


# # METHODS
def get_annots(folder):
    """
    Load manual annotations corresponding to "ground truth" of facial
    expressions produced during prediction phase
    """
    try:
        arr = np.load(join(folder, "annot.py"))
        res = True
    except FileNotFoundError:
        res = False
        arr = None

    return res, arr


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    sid = argv[1] if len(argv) > 1 else SUBJECT_ID
    subj_folder = inut.get_subj_folder(SRC_FOLDER, sid)

    # Get paths to previously recorded images and corresponding data
    annot_set, pred_test = inut.get_image_sets(subj_folder, 36)
    # "Regular" split is based on the learning and prediction phases as they
    # were performed during the experiment
    # Uncommment below to apply same split as run_fer_eval
    split = len(annot_set) * 3 // 4
    print(split)
    img_sets = (annot_set[:split], annot_set[split:][::-1])
    shuffle(img_sets[0])

    # FER model
    model = rec.ModelByState()
    # Keypoint extraction method and tools
    kp_tools, get_kp = imut.get_kp_method(KP_MODE, rec.N_KEYPOINTS)
    # Container for prediction results
    pred_results = []

    # # LOOPS
    for img_set, is_learning in zip(img_sets, (True, False)):
        for ind, cell in enumerate(img_set):
            if cell[1] == 0:
                continue
            # Load and process saved image
            frame = cv.imread(cell[0], cv.IMREAD_ANYCOLOR)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            kp_coord, kp_data = get_kp(gray, *kp_tools)
            if not kp_coord:
                print("No keypoints found!")
            curr_state = cell[1] if is_learning else None

            # Feed data to model
            model.clear_stm()
            img_kp = frame.copy()
            for kp_c, kp_vec in zip(kp_coord, kp_data):
                model.feed(kp_vec, curr_state)
                if is_learning:
                    model.update()
                elif SHOW:
                    # Show activities of LMS layer
                    lms_acts = [n.activity for n in model.lms.neurons]
                    print(np.round(lms_acts, 2))
                    # Show keypoint with color coding
                    col = [COLOR_CODES[np.argmax(lms_acts)]]
                    img_kp = imut.draw_kp(img_kp, [kp_c], col)

            if not is_learning:
                pred_vec, pred_emo = model.predict()
                pred_results.append((pred_emo, cell[1]))
                if SHOW:
                    print("Truth:", rec.PRIMARY_STATES[cell[1]])
                    print("Prediction:", np.round(pred_vec, 3),
                          rec.PRIMARY_STATES[pred_emo])
                    cv.imshow("Image with keypoints", img_kp)
                    char = cv.waitKey(-1) & 0xFF
                    if char == ord('q'):  # Exit program
                        raise KeyboardInterrupt
                    elif char == ord('s'):  # Skip to results
                        cv.destroyAllWindows()
                        SHOW = False

    recogs = [pred == truth for pred, truth in pred_results]
    recog_rate = int(np.mean(recogs) * 100)
    print(len(model.saw.neurons))
    conf_matrix = np.zeros([rec.N_STATES] * 2)
    for pred, truth in pred_results:
        conf_matrix[truth, pred] += 1
    print(conf_matrix)
    print(recog_rate)
    plut.plot_conf_mat(conf_matrix)
    # plt.savefig("img/{}_prim.png".format(sid), dpi=150)
