# -*- coding: utf-8 -*-
"""
Primary facial expressions recognition
With avatar, sliding stm in the test, using dlib, in real time
"""
# # IMPORTS
# - Built-in
import time
# - Third-party
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from imutils import face_utils
import dlib
# - Local
from lms import LMS
from saw import SAW
from utils.stm import STM


# # CONSTANTS
expressions = ["happy", "sad", "angry", "afraid", "surprise"]
# Resolution of images retrieved from camera
cols = 640  # 320 160
rows = 480  # 240 120
n_frames_rec = 10
learning_duration = 100

n_features = 128  # Number of feature descriptors used in a point of the image
n_points = 10     # Number of points used in an image

max_saw_size = 250    # Max size of a SAW
vigilance_saw = 0.98  # Vigilance threshold


# # METHODS
def multiply_lists(l1, l2):
    """
    Multiply two lists one to one
    """
    return np.multiply(l1, l2).tolist()


def get_input_vector(neurons, size=10):
    """
        Format neuron activities in a vector (list) of given size. If too many
        neurons, extra ones are discarded. If too few, vector is padded with
        zeros. The neuron activities must be previously calculated (see
        calculate_activity)

        Parameters :
            neurons: list of neurons
            size: size of output vector (default 10)
    """
    acts = [neu.activity for neu in neurons]
    n_diff = size - len(neurons)
    res = acts[:size] if n_diff <= 0 else acts + [0] * n_diff

    return res


def get_neurons_array(neurons):
    """
        Format neuron weightings as 2D array
        The neuron weightings must be previously calculated (see
        calculate_weightings)

        Parameters:
            neurons: matrix of neurons, as a list of lists
    """

    weights = [[neu.weighting for neu in row] for row in neurons]

    return np.array(weights)


# # MAIN
if __name__ == "__main__":
    # - PREPARE
    # Initialize HoG-based face detector
    detector = dlib.get_frontal_face_detector()
    # Load catalogue of pre-shaped models
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Initialize video capture and define image resolution
    cap = cv.VideoCapture(0)
    cap.set(3, cols)
    cap.set(4, rows)
    # norm = 1.0 / n_frames_rec

    saw_image = SAW(n_features,     # Dimension
                    vigilance_saw,  # Vigilance threshold
                    max_saw_size,   # Max amount of neurons
                    0.01)           # Learning rate

    lms_image = LMS(max_saw_size, 5, 0.1, range_min=-0.3, range_max=0.3)

    # STM for visual modality
    stm_image = STM(5,               # Number of neurons
                    1.0 / n_points,  # Norm
                    1.0)             # Oblivion coeff

    # STM for sliding over emotion changes
    stm_sliding = STM(5,    # Number of neurons
                      1.0,  # Norm
                      0.8)  # Oblivion coeff

    # - CAPTURE
    # Learning
    print("********* LEARNING for {}s *********\n".format(learning_duration))
    index = 0
    t_counter = 0
    t1 = time.perf_counter()
    t2 = t1
    while t2 - t1 < learning_duration:
        # After n_frames_rec images were captured, change facial expression
        if (t_counter % n_frames_rec) == 0:
            desired_output = [0, 0, 0, 0, 0]
            # index = randint(0, 4)
            index = (index + 1) % 5
            desired_output[index] = 1
            print("***** Supervision: {} *****".format(expressions[index]))
            print("Time elapsed: {:.3}s".format(t2 - t1))
            print(5)
            time.sleep(1)
            print(4)
            time.sleep(1)
            print(3)
            time.sleep(1)
            print(2)
            time.sleep(1)
            print(1)
            time.sleep(1)

        rects = []
        while len(rects) == 0:  # Loop until at least one face is found
            # Retrieve current image in grayscale
            ret, frame = cap.read()
            current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Attempt to detect a face
            rects = detector(current, 1)

        # Determine the face region’s shape (as an array of 2D coordinates)
        face_shape = predictor(current, rects[0])
        shape_arr = face_utils.shape_to_np(face_shape)

        # Selecting which keypoints we keep from 68 keypoints. See doc at:
        # https://emrah.fr/post/detection-de-reperes-faciaux-avec-dlib-opencv-et-python
        r_eyebrow = shape_arr[17:22, :]
        r_eyebrow = r_eyebrow[::2, :]
        l_eyebrow = shape_arr[22:27, :]
        l_eyebrow = l_eyebrow[::2, :]
        mouth = shape_arr[48:68, :]
        mouth = mouth[:12:2, :]
        shape = np.vstack((r_eyebrow, l_eyebrow, mouth))

        # Using SIFT to get angular values
        sift = cv.SIFT_create()
        coords = cv.KeyPoint_convert(shape.tolist())
        kp, inputs_image = sift.compute(current, coords, None)
        # Normalize angular values in input_samples image
        inputs_image /= 360  # [°]

        # Recognition
        stm_image.clearNetwork()  # Clear values in memory

        for point in range(n_points):
            saw_image.load_input(inputs_image[point])
            saw_image.calculate_nets()
            saw_image.calculate_average()
            saw_image.calculate_standard_deviation()
            saw_image.calculate_activities()  # Heavyside

            input_vector = get_input_vector(saw_image.neurons, max_saw_size)
            lms_image.calculate_weightings(input_vector)
            lms_image.calculate_activities()  # sigmoid
            lms_image.calculate_errors(desired_output)

            # Integrate activities
            input_vector = get_input_vector(lms_image.neurons, size=5)
            stm_image.integrate(input_vector)

            # Update weights
            lms_image.update_neurons()
            saw_image.update_neurons()

        # TODO: why does this happen out of the loop?
        output = multiply_lists(input_vector, stm_image.neurons)
        print("Number of neuron recruits image:", saw_image.nb_neurons)

        # Display
        # cv.imshow('Frame', frame)  # display the current frame
        current = cv.drawKeypoints(current, kp, 0,
                                   color=(0, 255, 255), flags=0)
        cv.imshow("Frame", current)
        if cv.waitKey(10) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            print("\nLearning interrupted\n")
            break

        t_counter += 1
        t2 = time.perf_counter()

    print(t_counter, " iterations")

    # Test
    print("********** Starting TEST... **********")
    print(3)
    time.sleep(1)
    print(2)
    time.sleep(1)
    print(1)
    time.sleep(1)

    # Prepare plot for avatar
    fig, ax = plt.subplots(1, 1)
    ax.set_xticklabels([])  # Hide x and y labels of subplot
    ax.set_yticklabels([])
    img = None

    t1 = time.perf_counter()
    while 1:
        # TODO: factor next 80 lines with previous loop
        ret, frame = cap.read()
        current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        rects = []
        while len(rects) == 0:  # Loop until at least one face is found
            # Retrieve current image in grayscale
            ret, frame = cap.read()
            current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Attempt to detect a face
            rects = detector(current, 1)

        # Determine the face region’s shape (as an array of 2D coordinates)
        face_shape = predictor(current, rects[0])
        shape_arr = face_utils.shape_to_np(face_shape)

        r_eyebrow = shape_arr[17:22, :]
        r_eyebrow = r_eyebrow[::2, :]
        l_eyebrow = shape_arr[22:27, :]
        l_eyebrow = l_eyebrow[::2, :]
        mouth = shape_arr[48:68, :]
        mouth = mouth[:12:2, :]
        shape = np.vstack((r_eyebrow, l_eyebrow, mouth))

        sift = cv.SIFT_create()
        coords = cv.KeyPoint_convert(shape.tolist())
        kp, inputs_image = sift.compute(current, coords, None)
        inputs_image /= 360  # [°]

        stm_image.clearNetwork()
        for point in range(n_points):
            saw_image.load_input(inputs_image[point])
            saw_image.calculate_nets()
            saw_image.calculate_average()
            saw_image.calculate_standard_deviation()
            saw_image.calculate_activities()

            input_vector = get_input_vector(saw_image.neurons, size=max_saw_size)
            lms_image.calculate_weightings(input_vector)
            lms_image.calculate_activities()

            input_vector = get_input_vector(lms_image.neurons, size=5)
            stm_image.integrate(input_vector)

        stm_sliding.slide(stm_image.neurons)
        output = multiply_lists(input_vector, stm_sliding.neurons)
        recog_exp = expressions[np.argmax(output)]

        # print("Recognition:" + expressions[np.argmax(output)])
        print("{} | {}".format(output, expressions[np.argmax(output)]))

        # Display current frame and corresponding avatar
        current = cv.drawKeypoints(current, kp, 0,
                                   color=(0, 255, 255), flags=0)
        cv.imshow("Frame", current)

        im = plt.imread("{}.png".format(recog_exp))
        plt.title("Recognized as:" + recog_exp)
        if img is None:
            img = plt.imshow(im)
        else:
            img.set_data(im)
        plt.pause(0.001)  # needed to display the figure

        if cv.waitKey(10) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            plt.close()
            print("\nTest stopped\n")
            break

    t2 = time.perf_counter()
    print("Elapsed test time: {:.4}".format(t2 - t1))

    cap.release()
    plt.close()
    cv.destroyAllWindows()
