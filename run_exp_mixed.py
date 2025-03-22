# -*- coding: utf-8 -*-
"""
Capture images of human faces imitating mixed emotions produced by robotic head
By Sébastien Mick rewrite by Jean Pierre KONDE MONK
"""
# # IMPORTS
# - Built-in
from os.path import join
from time import time
from pickle import load
# - Third-party
from numpy import argmax
import cv2 as cv
# - Local
import utils.robot as rob
import utils.improc as imut
import utils.info as inut


# # CONSTANTS
# Recording parameters
DATA_FOLDER = "data/mix"
SUBJECT_ID = "d9"
# Pre-built trial order
with open("utils/order_mix.pkl", 'rb') as ifile:
    TRIAL_ORDER = [lev for lev, _ in load(ifile)]
N_TRIALS = len(TRIAL_ORDER)
N_IMG_PER_TRIAL = 40
COUNTDOWN = 3  # [s]
REFRESH_PERIOD = 0.01  # Refresh period [s]
# Template for info to print before each trial
TRIAL_STR = "Trial {} / " + str(N_TRIALS)
# Naming template for images: subject_trial_batch_emotion_traitlevels.png
IMG_STR = "{}_{}_{}_{}_{}.png"


# # METHODS
def lev_to_str(lev):
    """
    Duplicate from bits in utils.recog
    """
    max_l = []
    for group in lev:
        max_l.append(argmax(group))
    res = "".join((str(it) for it in max_l))

    return res


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    # Data folder
    subj_folder = inut.make_subj_folder(DATA_FOLDER, SUBJECT_ID)

    # Video capture and processing
    cap = imut.prepare_capture()
    # Connection with robot
    robot = rob.RobotInterface()
    robot.goto_neutral()

    print("OpenCV version: {}".format(cv.version.opencv_version))
    print("* * * * LEARNING * * * *\n")

    # Setup for loop
    expe_ongoing = True
    # Setup for first trial
    trial_count = 0
    batch_count = 0

    levels = TRIAL_ORDER[0]
    print(TRIAL_STR.format(1))
    robot.move_traits(levels)
    inut.make_countdown(COUNTDOWN)
    input()
    prev_t = time() + 0.5  # Set stopwatch "ahead of time" to clear buffer

    # # LOOP
    while expe_ongoing:
        try:
            delta_t = time() - prev_t
            # Read frame from camera as often as possible to avoid buffering
            _, frame = cap.read()
            # Only process a fraction of captured frames, periodically
            if delta_t > REFRESH_PERIOD:
                # Save frame
                filename = IMG_STR.format(SUBJECT_ID,
                                          trial_count,
                                          batch_count,
                                          "na",
                                          lev_to_str(levels))
                cv.imwrite(join(subj_folder, filename), frame)
                cv.imshow("Image", frame)
                cv.waitKey(1)

                # Update batch count
                batch_count += 1
                # Full batch means trial is over
                if batch_count % N_IMG_PER_TRIAL == 0:
                    robot.goto_neutral()
                    batch_count = 0
                    trial_count += 1  # Begin next trial
                    # Continue learning phase
                    if trial_count < N_TRIALS:
                        inut.make_countdown(2)
                        levels = TRIAL_ORDER[trial_count]
                        robot.move_traits(levels)
                        print(TRIAL_STR.format(trial_count + 1))
                        inut.make_countdown(COUNTDOWN)
                        input()
                        # Changing facial traits is fast enough to
                        # end before countdown is over
                    else:
                        expe_ongoing = False

                prev_t = time()

        except KeyboardInterrupt:
            expe_ongoing = False

    cap.release()
    cv.destroyAllWindows()
    robot.shutdown()
