#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

"""
Run : python PFER_base.py
Without avatar, in real time
"""

from lms import *
from saw import *
from utils.stm import *
import numpy as np
import time
import cv2 as cv

print("OpenCV version :  {0}".format(cv.__version__))

def multiplyLists(l1, l2) :
    """
        Multiply two lists one to one
    """

    l3 = [0.0 for i in range(len(l1))]
    for i in range(len(l1)) :   #for each neurons of the network
        l3[i] = l1[i]*l2[i]
    return l3


def createInputVector(neurons, size=10) :
    """
        Function used in order to transform neurons' output in a vector
        The activity of the neurons must be previously calculated (see calculate_activity)

        Parameters :
        neurons -- the list of neurons
        size -- the size of desired vector (10 by default)
    """

    inputVector = []
    for i in range(size) :      #for each part of the vector
        if(i<len(neurons)) :    #if there are still neurons, put the neuron's activity
            inputVector.append(neurons[i].activity)
        else :                  #else put a null value
           inputVector.append(0)
    return inputVector


def neuronsToArray(neurons, rows, cols) :
    """
        Function used in order to transform neurons' output in a matrix
        The weighting of the neurons must be previously calculated (see calculate_weightings)

        Parameters :
        neurons -- the matrix of neurons
        rows -- the number of neurons of the network in rows
        cols -- the number of neurons of the network in columns
    """

    array = np.zeros((rows,cols))
    for i in range(rows) :        #for each neurons of the network in rows
        for j in range(cols) :    #for each neurons of the network in columns
            array[i][j] = neurons[i][j].weighting

    return array


'''
MAIN 
'''

expressions = ['tristesse', 'joie', 'surprise', 'neutre', 'colere']

cols = 320 # 640 160
rows = 240 # 480 120
nbFramesIntegrated = 10

#Initialize video capture
cap = cv.VideoCapture(0)
ret = cap.set(3, cols)
ret = cap.set(4, rows)

norm = 1.0/nbFramesIntegrated


nbFeatures = 128                              #The number of descriptor used in a point of the image
nbPoints = 10                                 #The number of point used in an image
sawSizeImage = 150                           #The maximum size of a SAW for the image

vigilanceSawImage = 0.98

sawImage = SAW(nbFeatures, vigilanceSawImage, sawSizeImage, 0.01)#Creation of a SAW(dimension, vigilance threshold, neurons maximum, learning rate)
lmsImage = LMS(sawSizeImage, 5, 0.1, range_min=-0.3,
               range_max=0.3)  #Creation of a LMS(dimension, number of neurons, learning rate)
stmImage = STM(5, 1.0/nbPoints)                         #Creation of a STM(number of neurons) for the visual modality

inputImage = []                        #The descriptors input(matrix)

desiredOutput = [0,0,0,0,0]

#Capture
t = 0
#tps1 = time.clock()
tps1 = time.perf_counter()
tps2 = tps1
duree_app = 100


######################################### Learning #########################################
print("*********** LEARNING during {} secs **************".format(duree_app))
index = 0
while (tps2-tps1) < duree_app:

    #Change of facial expression
    if (t%nbFramesIntegrated) == 0 :
        desiredOutput = [0,0,0,0,0]
        #index = randint(0, 4)
        index = (index+1)%5
        desiredOutput[index] = 1
        print("**" + expressions[index] + "**")
        nom = expressions[index] + ".png"
        matrice = cv.imread(nom)
        cv.imshow('Avatar',matrice)
        cv.waitKey(30)

        print(4)
        time.sleep(1)
        print(3)
        time.sleep(1)
        print(2)
        time.sleep(1)
        print(1)

    ######################################### GET DATA #########################################

    #Get video frames
    ret, frame = cap.read()                                 #get the image
    current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)         #put it in grey level

    sift = cv.SIFT_create(nbPoints)                     #create sift  with n_points keypoints
    kp, des = sift.detectAndCompute(current, None)          #get points and descriptors

    #orb = cv.ORB_create(n_points)
    #kp, des = orb.detectAndCompute(current,None)


    inputImage = des                          #add descriptors matrix
    #inputs_image = [list(map(float, l)) for l in des]
    #print(len(inputs_image))
    #print(len(inputs_image[0]))

    #Normalization of the input_samples image (have data between 0 and 1)
    for j in range(nbFeatures) :
        for i in range(len(inputImage)) :
            inputImage[i][j] = inputImage[i][j]/360

        ######################################### RECOGNITION #########################################

    #If this not the first frame
    if t != 0 :

        stmImage.clearNetwork()                      #clear memory value

        #for each points of interest
        for point in range(nbPoints) :
            #load the data
            sawImage.load_input(inputImage[point])

            #Calculate saw's activity
            sawImage.calculate_nets()
            sawImage.calculate_average()
            sawImage.calculate_standard_deviation()
            sawImage.calculate_activities()#heavyside

            #Calculate lms's activity
            inputVector = createInputVector(sawImage.neurons, size=sawSizeImage)
            lmsImage.calculate_weightings(inputVector)
            lmsImage.calculate_activities() #sigmoide
            lmsImage.calculate_errors(desiredOutput)

            #Integrate activities
            inputVector = createInputVector(lmsImage.neurons, size=5)
            stmImage.integrate(inputVector)

            #Update weights

            lmsImage.update_neurons()
            sawImage.update_neurons()

            output = multiplyLists(inputVector, stmImage.neurons)
        #print(output)
        print("Group SAW: Number of neuron recruits: ",sawImage.nb_neurons)


        ######################################### DISPLAY and SAVE #########################################

        #cv.imshow('Frame1',frame)                                                             #display the current frame
        current=cv.drawKeypoints(current,kp, 0,color=(0,255,255), flags=0)
        cv.imshow('Frame2',current)

        if cv.waitKey(1) & 0xFF == ord('q'):                                                 #wait to visualize
            print("\nLearning stopped\n")
            break

    t += 1

    tps2 = time.perf_counter()

print(t, " iterations ")


######################################### Test #########################################
print("******** Starting TEST... ********")
print(3)
time.sleep(1)
print(2)
time.sleep(1)
print(1)
time.sleep(1)

tps1 = time.perf_counter()

while 1 :

    #Get video frames
    ret, frame = cap.read()                                 #get the image
    current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)         #put it in grey level

    sift = cv.SIFT_create(nbPoints)                     #create sift  with n_points keypoints
    kp, des = sift.detectAndCompute(current, None)          #get points and descriptors

    inputImage = des                          #add descriptors matrix

    #Normalization of the input_samples image (have data between 0 and 1)
    for j in range(nbFeatures) :
        for i in range(len(inputImage)) :
                inputImage[i][j] = inputImage[i][j] / 360    #normalize


    stmImage.clearNetwork()                      #clear memory value
    #for each points of interest
    for point in range(nbPoints) :
        #load the data
        sawImage.load_input(inputImage[point])

        #Calculate saw's activity
        sawImage.calculate_nets()
        sawImage.calculate_average()
        sawImage.calculate_standard_deviation()
        sawImage.calculate_activities() #heavyside

        #Calculate lms's activity
        inputVector = createInputVector(sawImage.neurons, size=sawSizeImage)
        lmsImage.calculate_weightings(inputVector)
        lmsImage.calculate_activities() #sigmoide

        #Integrate activities
        inputVector = createInputVector(lmsImage.neurons, size=5)
        stmImage.integrate(inputVector)
        output = multiplyLists(inputVector, stmImage.neurons)

    #print("Recognition : " + expressions[np.argmax(output)])
    print("{} - {}".format(output,expressions[np.argmax(output)]) )

    ######################################### DISPLAY #########################################

    #cv.imshow('Frame',frame)                                                             #display the current frame
    current=cv.drawKeypoints(current,kp, 0,color=(0,255,255), flags=0)
    cv.imshow('Frame',current)

    if cv.waitKey(1) & 0xFF == ord('q'):                                                 #wait to visualize
        print("\nTest stopped\n")
        break


tps2 = time.perf_counter()
print(tps2 - tps1,"temps de test écoulé\n")


#Close video capture
cap.release()
cv.destroyAllWindows()
