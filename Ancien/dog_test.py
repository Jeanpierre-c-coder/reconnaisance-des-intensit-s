#!/usr/bin/env python
# * UTF-8*

""" Test file for repmplace sift by dog  """

from time import sleep
import dog
import cv2 as cv
# basic expressions can be used or detected.
COLS = 320  # 640 160
ROWS = 240  # 480 120
NB_FRAMES_INTEGRATED = 10
NB_FEATURES = 128
# The number of descriptor used in a point of the image
NB_POINTS = 10
# The number of point used in an image
SAW_SIZE_IMAGE = 150
# The maximum size of a SAW for the image
VIGILANCE_SAW_IMAGE = 0.98
# The vigilance threshold of SAW for the image


# Initialize video capture
cap = cv.VideoCapture(0)
ret = cap.set(3, COLS)
ret = cap.set(4, ROWS)

ret, frame = cap.read()  # get the image
frame2 = cv.resize(frame,(240,240))
current = cv.cvtColor(frame2,
                          cv.COLOR_BGR2GRAY)
print(current)
# différence de 3 minimun affin de permettre que le filtre soit différent q
filter  = dog.generate_DoG_filter(120,4,1)
print("le filtre",filter[1])
test = dog.process_dog_from_image(current,0.5,0.2,filter)

test  = dog.compet_ptc_image(test,1)
point = dog.extract_keypoint_promethe(test, 5)
print(point) 
i=0 
"""for x in range (len(point)):
    cv.KeyPoint(point[x],10)
current = cv.drawKeypoints(frame2,
            cv.KeyPoint,
            0,
            color=(0, 255, 255),
            flags=0)
cv.imshow("frame",frame2)
"""

def Dog_image(input_image):
    frame2 = cv.resize(frame,(240,240))
    current = cv.cvtColor(frame2,
                          cv.COLOR_BGR2GRAY)
    filter  = dog.generate_DoG_filter(120,1,0.1)
    test = dog.process_dog_from_image(current,0.5,0.2,filter)
    return test