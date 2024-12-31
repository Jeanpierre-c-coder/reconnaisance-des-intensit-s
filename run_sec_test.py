# -*- coding: utf-8 -*-
"""
Run experiment with model learning to recognize different states of groups of
facial traits
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from os.path import join
from time import time, sleep
from pickle import load
# - Third-party
import numpy as np
import cv2 as cv
# - Local
import utils.recog as rec
import utils.robot as rob
import utils.improc as imut
import utils.info as inut


# # CONSTANTS
# Recording parameters
DATA_FOLDER = "data/seco"
SUBJECT_ID = "d9"
# Experiment parameters
EXP_MODE = 3
KP_MODE = "dog"  # {"dog", "dlib", "sift"}
# Template for files containing pre-built trial orders
with open("utils/order_elem.pkl", 'rb') as ifile:
    LEARN_ORDER = load(ifile)
N_LEARN_TRIALS = len(LEARN_ORDER)
N_IMG_PER_TRIAL = 30
N_LEARN_IMG = 15
COUNTDOWN = 3  # [s]
LEARN_PERIOD = 0.03  # Refresh periods in seconds
TEST_PERIOD = 0.5
# Template for info to print before each trial
TRIAL_STR = "Trial {} / " + str(N_LEARN_TRIALS) + " | {}"
# Naming template for images: subject_trial_batch_emotion_traitlevels.png
IMG_STR = "{}_{}_{}_{}_{}"


# # METHODS


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    # Data folder
    subj_folder = inut.make_subj_folder(DATA_FOLDER, SUBJECT_ID)
    # Info file
    inut.write_info_file(subj_folder, inut.make_info_dic(EXP_MODE, KP_MODE))

    # Video capture and processing
    cap = imut.prepare_capture()
    # Image processing mode
    kp_tools, get_kp = imut.get_kp_method(KP_MODE, rec.N_KEYPOINTS)
    # FER model and corresponding trial order
    model = rec.ModelThreeway() if EXP_MODE > 2 else rec.ModelByGroup()
    # Connection with robot
    robot = rob.RobotInterface()
    robot.goto_neutral()

    print("OpenCV version: {}".format(cv.version.opencv_version))
    print("* * * * LEARNING * * * *\n")

    # Setup for recording
    pred_results = []

    # Setup for loop
    expe_ongoing = True
    update_period = LEARN_PERIOD
    n_img_to_save = N_IMG_PER_TRIAL
    n_img_to_process = N_LEARN_IMG
    learning_ongoing = True
    # Setup for first trial
    trial_c = 0
    batch_c = 0
    emo, levels, emo_str, lev_str, grp = rec.unpack_seco(LEARN_ORDER[0],
                                                         EXP_MODE)
    print(TRIAL_STR.format(1, emo_str))
    robot.move_traits(levels)
    sleep(COUNTDOWN)
    prev_t = time() + 0.5  # Set stopwatch "ahead of time" to clear buffer

    # # LOOP
    while expe_ongoing:
        try:
            delta_t = time() - prev_t
            # Read frame from camera as often as possible to avoid buffering
            _, frame = cap.read()

            # Only process a fraction of captured frames, periodically
            if delta_t < update_period:
                continue

            # Save frame on disk
            info = [SUBJECT_ID, trial_c, batch_c,
                    emo_str, lev_str]
            if EXP_MODE > 1:
                info.append(grp)
            filename = "_".join((str(i) for i in info)) + ".png"
            cv.imwrite(join(subj_folder, filename), frame)
            batch_c += 1
            prev_t = time()

            # Only process a fraction of saved images
            if batch_c <= n_img_to_process:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                kp_coord, kp_data = get_kp(gray, *kp_tools)
                # If too few keypoints were found, skip this iteration
                if kp_data.shape[0] < rec.N_KEYPOINTS:
                    print("Not enough keypoints!")
                    # Skip this frame and process another
                    batch_c -= 1
                    continue
                # Before feeding data from new frame, clear local STM
                model.clear_stm()
                # For each keypoint, feed data to FER model
                for kp_vec in kp_data:
                    if EXP_MODE > 1 and learning_ongoing:
                        model.feed_select(kp_vec, levels, grp)
                    else:
                        model.feed(kp_vec, levels)
                    if learning_ongoing:
                        model.update()
                img_kp = cv.drawKeypoints(frame, kp_coord, 0, flags=0,
                                          color=(0, 255, 255))
                cv.imshow("Image with keypoints", img_kp)
                cv.waitKey(1)

            # Begin next trial if batch is full
            if batch_c < n_img_to_save:
                continue
            batch_c = 0
            if learning_ongoing:
                trial_c += 1
                robot.goto_neutral()
                # Continue learning phase
                if trial_c < N_LEARN_TRIALS:
                    sleep(1)
                    info = rec.unpack_seco(LEARN_ORDER[trial_c], EXP_MODE)
                    emo, levels, emo_str, lev_str, grp = info
                    robot.move_traits(levels)
                    print(TRIAL_STR.format(trial_c + 1, emo_str))
                    # Changing traits is fast enough to end before countdown
                    prev_t = time() + COUNTDOWN
                # Switch to prediction phase
                else:
                    learning_ongoing = False
                    trial_c = -1
                    update_period = TEST_PERIOD
                    n_img_to_save = 1
                    n_img_to_process = 1
                    levels, emo, grp = None, None, None
                    emo_str, lev_str = "na", "na"
                    if EXP_MODE > 2:
                        print([len(saw.neurons) for saw, _ in model.ways])
                    else:
                        print(len(model.saw.neurons))
                    print("\n* * * * TESTING * * * *\n")
                    prev_t = time() + 5
            # Continue prediction phase
            else:
                trial_c -= 1  # Negative trial count in prediction
                # Make prediction based on current frame
                pred_v, pred_i = model.predict(output="pred")
                print("Output vector:", np.round(pred_v, 3))
                print("Prediction:", rec.PRIMARY_STATES[pred_i])
                pred_l = rec.reshape_as_levels(pred_v)
                # Move robot's facial traits accordingly
                robot.move_traits(pred_l)

        except KeyboardInterrupt:
            expe_ongoing = False

    cap.release()
    cv.destroyAllWindows()
    robot.shutdown()
