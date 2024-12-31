# -*- coding: utf-8 -*-
"""
Image processing toolbox for keypoint extraction: Difference of Gaussians (DoG)
filtering and log-polar transform
Adapted by Sébastien Mick from a toolbox by Sylvain Colomer
"""
# # IMPORTS
# - Built-in
from math import pi, log
# - Third-party
import numpy as np
import cv2 as cv
import dlib
# from imutils import face_utils
# noinspection PyUnresolvedReferences
import cv2.ximgproc as ximproc  # Install opencv-contrib to include this module
# - Local


# # CONSTANTS
# Image processing
IMG_WIDTH = 320
IMG_HEIGHT = 240
# Resources for face detection
FACE_LANDMARKS_CAT = "utils/shape_predictor_68_face_landmarks.dat"
LANDMARK_SUBSET = np.array([17, 19, 21, 22, 24, 26, 48, 50, 52, 54, 56, 58])
# Alternative landmarks on the lips: 51, 54, 57
# DoG parameters
DOG_HALFSIDE = 15
DOG_S1, DOG_S2 = 3., 6.
# Polar transform parameters
POLAR_RADIUS = 30    # Radius of outer circle
POLAR_EXC = 4        # Radius of inner circle to exclude
POLAR_SIZE = (9, 9)  # Output array size
# Deriche gradiant parameters
GRAD_ALPHA, GRAD_OMEGA = 0.5, 0.6
EXC_WIDTH = 10  # Width of exclusion zones on borders of frame
# Radius for median filter
FILT_RADIUS = 25
# Radius for local maxima extraction
MAXIMA_RADIUS = 32
# Offset to skip the N first keypoints detected with DoG pipeline
KP_OFFSET = 0

# Prepare offset window for polar warp
if 0 < POLAR_EXC < POLAR_RADIUS:
    LOG_RATIO = log(POLAR_EXC) / log(POLAR_RADIUS)
    POLAR_OFFSET = int(POLAR_SIZE[0] * (1 / (1 - LOG_RATIO) - 1))
    POLAR_ZONE = (POLAR_SIZE[0] + POLAR_OFFSET, POLAR_SIZE[1])
else:
    POLAR_OFFSET, POLAR_ZONE = 0, POLAR_SIZE


# # METHODS
def prepare_capture():
    """
    Create and configure a VideoCapture object to retrieve frames from camera
    """
    cap = cv.VideoCapture("/dev/video0", cv.CAP_V4L)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)    # Set lower resolution
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)  # for faster capture
    # If using composite camera & USB capture card, set adequate pixel format
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("Y", "U", "Y", "V"))

    return cap


def get_kp_method(mode, n_kp):
    """
    Prepare CV tools and keypoint extraction method corresponding to given mode
    """
    if mode == "dog":
        tools = (get_dog_tools(), n_kp)
        kp_method = get_keypoints_dog
    elif mode == "dlib":
        face_det, shape_pred = get_dlib_tools()
        tools = (face_det, shape_pred)
        kp_method = get_keypoints_dlib
    else:  # SIFT
        sift_tool = cv.SIFT_create(n_kp)
        tools = (sift_tool, )
        kp_method = get_keypoints_sift

    return tools, kp_method


def build_dog_kernel(halfside, sig1, sig2):
    """
    Build 2D convolution kernel with a Difference of Gaussians (DoG)
    approximating a "Mexican hat" kernel

    halfside: half-length of kernel side, as an integer
    sig1: standard deviation of the narrow Gaussian, as a float
    sig2: standard deviation of the wide Gaussian, as a float
    """
    # Prepare constant factors involved in computation of each Gaussian bell
    two_s1_sq = 2. * sig1 ** 2
    two_s2_sq = 2. * sig2 ** 2
    k1 = 1 / (pi * two_s1_sq)
    k2 = 1 / (pi * two_s2_sq)

    # Build square grid with odd number of rows
    side_vec = np.arange(-halfside + 1, halfside)
    grid_x, grid_y = np.meshgrid(side_vec, side_vec, sparse=True)
    rad_sq = grid_x ** 2 + grid_y ** 2
    ker = k1 * np.exp(-rad_sq / two_s1_sq) - k2 * np.exp(-rad_sq / two_s2_sq)
    # Using meshgrid and vector operations is faster than using loops

    return ker


def get_local_maxima(src, radius, n_max):
    """
    From a single-channel image, extract a given number of maximum values,
    expected to be local maxima. Returns an image containing only extracted
    maxima, and a list of their coordinates

    src: source image, as a single-channel matrix
    radius: radius of circular neighborhood to exclude around each previously
        extracted maximum
    n_max: number of maxima to extract, as an integer
    """
    loc = src.copy()  # Work on local copy of source image
    coord = []
    for _ in range(n_max):
        # Find location of global maximum value and extract it
        i_max, j_max = np.unravel_index(np.argmax(loc, axis=None), loc.shape)
        # Axes are reversed between NumPy and OpenCV conventions
        coord.append((int(j_max), int(i_max)))
        # Exclude circular neighborhood from next global maximum detection
        cv.circle(loc, (j_max, i_max), radius, color=0, thickness=-1)

    return coord


def get_kp_regions_polar(src, kp_coord):
    """
    Extract keypoint regions of given size in polar domain

    src: source image
    kp_coord: keypoint coordinates, as a list of pairs of integers
    out_size: size of output to extract from log-polar space, as an integer or
        pair of integers
    polar_radius: radius of circular region around each keypoint to warp in
        log-polar space
    """
    res = []
    for kp_loc in kp_coord:
        reg = cv.warpPolar(src, POLAR_ZONE, kp_loc, POLAR_RADIUS,
                           cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG)
        res.append(reg[:, POLAR_OFFSET:].ravel())

    # Normalize output of log-polar transform
    return np.array(res, dtype=float) / 255


def conv_edge_with_kernel(src, ker, alpha, omega, exc_width, filt_rad):
    """
    Compute gradient map from source image with Deriche filter, then convolve
    with given kernel

    src: source image, as a single-channel matrix
    ker: convolution kernel, as a single-channel matrix
    alpha: smoothing parameter for Deriche gradient filter (between 0.1 and 5)
    omega: second parameter for Deriche gradient filter (lower than 1)
    exc_width: width of zone to exclude along each border of output image
    """
    # Compute Deriche gradient
    src_f = src.astype(np.float32)
    deriche_x = ximproc.GradientDericheX(src_f, alpha, omega)
    deriche_y = ximproc.GradientDericheY(src_f, alpha, omega)
    grad = (deriche_x ** 2 + deriche_y ** 2) ** 0.5

    # Exclude borders to counter border highlighting effect
    grad[np.arange(-exc_width, exc_width), :] = 1.
    grad[:, np.arange(-exc_width, exc_width)] = 1.
    grad_n = cv.normalize(grad, 0, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    # Convolve median blur with kernel
    filt = cv.medianBlur(grad_n, filt_rad, 0)
    conv = cv.filter2D(filt, cv.CV_32F, ker, borderType=cv.BORDER_REPLICATE)

    conv_n = cv.normalize(conv, 0, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # cv.imshow("grad", grad_n)
    # cv.imshow("filt", filt)
    # cv.imshow("conv", conv_n)

    return conv_n


def get_keypoints_dog(src, ker, n_keypoints, mask=None):
    """
    Image processing pipeline: build DoG kernel, compute and convolute with
    edge map, find local maxima and extract neighborhoods with log-polar
    transform

    src: source image, in grayscale
    ker: DoG kernel
    n_keypoints: number of local maxima to extract
    """
    conv = conv_edge_with_kernel(src, ker, alpha=GRAD_ALPHA, omega=GRAD_OMEGA,
                                 exc_width=EXC_WIDTH, filt_rad=FILT_RADIUS)
    if mask in (0, 1):
        halfh = IMG_HEIGHT // 2
        if mask == 0:
            conv[:halfh, :] = 0.
        else:
            conv[halfh:, :] = 0.
    coord = get_local_maxima(conv, radius=MAXIMA_RADIUS,
                             n_max=n_keypoints + KP_OFFSET)[KP_OFFSET:]

    data = get_kp_regions_polar(src, coord)
    coord_kp = cv.KeyPoint_convert(coord)

    return coord_kp, data


def get_dog_tools():
    """
    Return kernel with typical parameters for facial trait extraction
    """
    ker = build_dog_kernel(DOG_HALFSIDE, DOG_S1, DOG_S2)

    return ker


def get_keypoints_dlib(src, detector, predictor):
    """
    Image processing pipeline: apply HoG-based face detector, predict locations
    of facial landmarks and select a subset of them, corresponding to eyebrows
    and mouth

    src: source image, in grayscale
    detector: face detector
    predictor: pre-trained facial landmark detector
    """
    # Attempt to detect a face in the frame
    rects = detector(src, 1)  # Result is a list of rectangular bounding boxes
    if len(rects) == 0:
        return [], np.zeros((1, 1))

    # Predict shape of found face as a set of facial landmarks
    face_shape = predictor(src, rects[0])  # Only one face should be found
    shape_arr = face_utils.shape_to_np(face_shape)  # Contains 68 landmarks
    # Select a subset of landmarks out of 68
    coord = shape_arr[LANDMARK_SUBSET, :].tolist()
    # Format as list of CV keypoints
    coord_kp = cv.KeyPoint_convert(coord)
    data = get_kp_regions_polar(src, coord)

    return coord_kp, data


def get_dlib_tools():
    """
    Return face detector and pre-trained shape predictor from dlib
    """
    # HoG-based face detector
    det = dlib.get_frontal_face_detector()
    # Shape predictor with catalogue of face landmarks
    pred = dlib.shape_predictor(FACE_LANDMARKS_CAT)

    return det, pred


def get_keypoints_sift(src, sift):
    """
    Image processing pipeline: find SIFT keypoints and compute corresponding
    descriptors

    src: source image, in grayscale
    sift: OpenCV's SIFT object
    """
    coord, data = sift.detectAndCompute(src, None)

    return coord, data / 360  # Normalize


# # EXTRA
def draw_kp(src, kp_list, colors, text=False):
    """
    Draw circles highlighting keypoint locations in given image

    src: source image
    kp_list: keypoints
    color: color code of drawn circles
    """
    res = src.copy()
    cols = [colors] * len(kp_list) if type(colors[0]) is int else colors

    for ind, (kp, col) in enumerate(zip(kp_list, cols)):
        coord = [int(co) for co in kp.pt]
        cv.circle(res, coord, POLAR_RADIUS, col, 1, lineType=cv.LINE_AA)
        if text:
            cv.putText(res, str(ind), coord, cv.FONT_HERSHEY_SIMPLEX, 0.4,
                       color=col, thickness=1)

    return res


def get_shape_diff(img1, img2):
    """
    Get padding dimensions to fit one image to the size of the other
    """
    hdiff = abs(img1.shape[0] - img2.shape[0])
    wdiff = abs(img1.shape[1] - img2.shape[1])
    top = hdiff // 2
    bottom = top if hdiff % 2 == 0 else top + 1
    left = wdiff // 2
    right = left if wdiff % 2 == 0 else left + 1

    return top, bottom, left, right


def get_circ_weighting(shape, min_val=0.9):
    """
    Build a 2D filter of given shape containing circularly distributed
    weightings
    """
    (cx, x), (cy, y) = ((dim // 2, np.arange(dim)) for dim in shape)
    xx, yy = np.meshgrid(x - cx, y - cy, sparse=True)
    rad = (xx ** 2 + yy ** 2) ** 0.5
    rad_max = np.max(rad)
    res = (min_val - 1) / rad_max * rad + 1

    return res.T
