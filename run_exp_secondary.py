# -*- coding: utf-8 -*-
"""
Run experiment with model learning to recognize activity levels of groups of
facial traits from elementary expressions
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from os.path import join
from time import time
from pickle import load
# - Third-party
import numpy as np
# import cv2 as cv
# - Local
import utils.recog as rec
import utils.robot as rob
import utils.improc as imut
import utils.info as inut


# # CONSTANTS
# Recording parameters
DATA_FOLDER = "data/seco"
SUBJECT_ID = "c8"
# Experiment parameters
EXP_MODE = 3
KP_MODE = "dog"  # {"dog", "dlib", "sift"}
# Template for files containing pre-built trial orders
with open("utils/order_mode{}.pkl".format(EXP_MODE), 'rb') as ifile:
    LEARN_ORDER = load(ifile)
N_LEARN_TRIALS = len(LEARN_ORDER)
N_IMG_PER_TRIAL = 15
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
    model = rec.ModelByGroup()
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
    learning_ongoing = True
    frame_ok = True
    # Setup for first trial
    trial_count = 0
    batch_count = 0
    emo, levels, emo_str, lev_str, grp = rec.unpack_seco(LEARN_ORDER[0],
                                                         EXP_MODE)
    state = levels
    print(TRIAL_STR.format(1, emo_str))
    robot.move_traits(levels)
    inut.make_countdown(5)
    prev_t = time() + 0.5  # Set stopwatch "ahead of time" to clear buffer

    # # LOOP
    while expe_ongoing:
        try:
            delta_t = time() - prev_t
            # Read frame from camera as often as possible to avoid buffering
            _, frame = cap.read()
            # Only process a fraction of captured frames, periodically
            if delta_t > update_period:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                kp_coord, kp_data = get_kp(gray, *kp_tools)

                # If too few keypoints were found, skip this iteration
                if kp_data.shape[0] < rec.N_KEYPOINTS:
                    if frame_ok:  # Warn user if not already done
                        frame_ok = False
                        print("Not enough keypoints!")
                else:
                    frame_ok = True

                    # Save frame
                    info = [SUBJECT_ID, trial_count, batch_count,
                            emo_str, lev_str]
                    if EXP_MODE > 1:
                        info.append(grp)
                    filename = "_".join((str(i) for i in info)) + ".png"
                    cv.imwrite(join(subj_folder, filename), frame)

                    # Before feeding data from new frame, clear local STM layer
                    model.clear_stm()
                    # For each keypoint, feed data to FER model
                    for kp_vec in kp_data:
                        if EXP_MODE == 2 and learning_ongoing:
                            model.feed_select(kp_vec, state, grp)
                        else:
                            model.feed(kp_vec, state)
                        if learning_ongoing:
                            model.update()

                    img_to_show = cv.drawKeypoints(frame, kp_coord, 0, flags=0,
                                                   color=(255, 255, 0))
                    cv.imshow("Image with keypoints", img_to_show)
                    cv.waitKey(1)

                    if learning_ongoing:
                        # Update batch and trial count
                        batch_count += 1
                        # Full batch means trial is over
                        if batch_count % N_IMG_PER_TRIAL == 0:
                            robot.goto_neutral()
                            batch_count = 0
                            trial_count += 1  # Begin next trial
                            # Continue learning phase
                            if trial_count < N_LEARN_TRIALS:
                                inut.make_countdown(1)
                                info = rec.unpack_seco(LEARN_ORDER[trial_count],
                                                       EXP_MODE)
                                emo, levels, emo_str, lev_str, grp = info
                                state = levels
                                robot.move_traits(levels)
                                print(TRIAL_STR.format(trial_count + 1, emo_str))
                                inut.make_countdown(COUNTDOWN)
                                # Changing facial traits is fast enough to
                                # end before countdown is over
                            # Switch to prediction phase
                            else:
                                learning_ongoing = False
                                trial_count = -1
                                update_period = TEST_PERIOD
                                emo, emo_str, lev_str = None, "na", "na"
                                state = None
                                print(len(model.saw.neurons))
                                print("\n* * * * TESTING * * * *\n")
                                inut.make_countdown(5)
                    else:
                        trial_count -= 1  # Negative trial count in prediction
                        # Make a prediction based on current frame
                        pred_v, pred_i = model.predict(output="pred")
                        print("Output vector:", np.round(pred_v, 3))
                        print("Prediction:", rec.PRIMARY_STATES[pred_i])
                        pred_l = rec.reshape_as_levels(pred_v)
                        # Move robot's facial traits accordingly
                        robot.move_traits(pred_l)

                prev_t = time()

        except KeyboardInterrupt:
            expe_ongoing = False

    cap.release()
    cv.destroyAllWindows()
    robot.shutdown()
