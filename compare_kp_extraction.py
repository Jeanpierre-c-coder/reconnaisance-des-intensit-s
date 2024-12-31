# -*- coding: utf-8 -*-
"""
Compare various keypoint extraction methods offline
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from os import listdir
from os.path import join, isfile
# - Third-party
import cv2 as cv
# - Local
from utils.info import get_subj_folder
import utils.improc as imut
import utils.recog as rec

# # CONSTANTS
# Recording parameters
SRC_FOLDER = "data/seco"
SUBJECT_ID = "b0"
SHOW_POLAR = True
SHOW_NUMBERS = False
SHOW_RADIUS = True


# # METHODS
def get_image_set(sid=None):
    """
    Get a set of images (as a list of paths) located in subject folder
    """
    folder = SRC_FOLDER if sid is None else get_subj_folder(SRC_FOLDER, sid)
    okfiles = [path for path in listdir(folder) if
               path.endswith(".png") and
               isfile(join(folder, path))]

    if len(okfiles) > 0:
        res = [join(folder, path) for path in okfiles]
    else:
        raise ValueError("No suitable image found in folder")

    return res


# # MAIN
if __name__ == "__main__":
    # # PREPARE
    # Get paths to previously recorded images
    img_set = get_image_set(SUBJECT_ID)
    # Prepare tool for SIFT-based method
    sift_tool = cv.SIFT_create(rec.N_KEYPOINTS)
    # Prepare tools for HOG-based method
    face_det, shape_pred = imut.get_dlib_tools()
    # Prepare kernel for DoG-based method
    ker = imut.get_dog_tools()

    # # LOOP
    for img in img_set:
        print(img.split("/")[-1])  # Filename can include relevant data
        frame = cv.imread(img, cv.IMREAD_ANYCOLOR)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Get keypoints with all methods
        dog_kp, _ = imut.get_keypoints_dog(gray, ker, rec.N_KEYPOINTS, None)
        dlib_kp, dlib_data = imut.get_keypoints_dlib(gray, face_det, shape_pred)
        sift_kp, _ = imut.get_keypoints_sift(gray, sift_tool)
        # Highlight them in source image
        with_dog = cv.drawKeypoints(frame, dog_kp, 0, color=(0, 0, 255),
                                    flags=0)
        if not SHOW_RADIUS:
            with_dlib = cv.drawKeypoints(with_dog, dlib_kp, 0, color=(0, 255, 0),
                                         flags=0)
            with_all = cv.drawKeypoints(with_dlib, sift_kp, 0, color=(255, 0, 0),
                                        flags=0)
        else:
            with_all = imut.draw_kp(with_dog, dog_kp, (200, 0, 55), SHOW_NUMBERS)

        if SHOW_POLAR and dlib_kp:
            for ind, kp_vec in enumerate(dlib_data):
                vign = kp_vec.reshape(imut.POLAR_SIZE)
                vign_n = cv.normalize(vign, 0, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                vign_up = cv.resize(vign_n, imut.POLAR_SIZE,
                                    interpolation=cv.INTER_LINEAR)
                cv.imshow("vign" + str(ind), vign_up)
        cv.imshow("Frame", with_all)
        char = cv.waitKey(-1) & 0xFF
        if char == ord('q'):
            break
    cv.destroyAllWindows()
