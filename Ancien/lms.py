#!/usr/bin/env python

"""
Contains the classes and the functions needed in order to create a Least Mean Square(LMS) neural network

Developped by Raphael BERGOIN
"""

from random import *
from math import *
import string
import numpy as np
import codecs
import random
import warnings


class LMS :
    """
        Class to represent a Least Mean Square(LMS) neural network
    """

    def __init__(self, dimension=1, nb_neurons=1, epsilon=0.1, rangeMin=-1, mangeMax=1) :
        """
            Constructor of the class LMS
        
            Parameters : 
            dimension -- the dimension of the neurons (1 by default)
            nb_neurons -- the number of neurons of the network (1 by default)
            epsilon -- the learning rate (0.1 by default)
            rangeMin -- the minimum initial value for a neuron's weight (-1 by default)
            mangeMax -- the maximal initial value for a neuron's weight (1 by default)
        """
        
        self.nb_neurons = nb_neurons
        self.dimension = dimension
        self.epsilon = epsilon
        self.rangeMin = rangeMin
        self.mangeMax = mangeMax
        self.neurons = [Neuron(dimension, rangeMin, mangeMax) for i in range(nb_neurons)]
        self.error = 1.0


    def calculate_weightings(self, data) :
        """
            Calculate the weighting of each neuron of the network
        
            Parameters : 
            data -- the data input vector given to the neuron
        """
        
        for i in range(self.nb_neurons) :    #for each neurons of the network
            self.neurons[i].load_input(data)
            self.neurons[i].calculate_weighting()

    
    def calculate_activities(self) :
        """
            Calculate the activity of each neuron of the network
            The weighting of each neuron of the network must be previously calculated (see calculate_weightings)
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            self.neurons[i].calculate_activity()
            
            
    def display_activities(self) :
        """
            Display the activity of each neuron of the network
            The activity of each neuron of the network must be previously calculated (see calculate_activities)
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            print(self.neurons[i].activity)

    def display_weights(self) :
        """
            Display the weight of each neuron of the network
            
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            print(self.neurons[i].w)


    def calculate_errors(self, desired_outputs) :
        """
            Calculate the local error of each neuron of the network
            The activity of each neuron of the network must be previously calculated (see calculate_activities)
        
            Parameters : 
            desired_outputs -- the desired outputs vector for each neuron
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            self.neurons[i].calculate_error(desired_outputs[i])
            
            
    def calculate_global_mean_error(self) :
        """
            Calculate the global mean error of the network
            The error of each neuron of the network must be previously calculated (see calculate_errors)
        """
        
        total = 0.0
        for i in range(self.nb_neurons) :   #for each neurons of the network
            total += abs(self.neurons[i].error)
        self.error = total/self.nb_neurons


    def update_neurons(self) :
        """
            Update the neurons' weights of the network
            The activity of each neuron of the network must be previously calculated (see calculate_activities)
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            self.neurons[i].update_weights(self.epsilon)
    


class Neuron :
    """
        Class to represent a simple neuron for a Least Mean Square(LMS) neural network
    """

    def __init__(self, dimension=1, rangeMin=-1, mangeMax=1) :
        """
            Constructor of the class Neuron
        
            Parameters : 
            dimension -- the dimension of the neurons (1 by default)
            rangeMin -- the minimum initial value for a neuron's weight (0 by default)
            mangeMax -- the maximal initial value for a neuron's weight (1 by default)
        """
        
        self.dimension = dimension
        self.w = [random.uniform(rangeMin, mangeMax) for i in range(dimension)]


    def load_input(self, data) :
        """
            Load an input in the neuron
        
            Parameters : 
            data -- the data input vector given to the neuron
        """
        
        self.input = data


    def calculate_weighting(self) :
        """
            Calculate the weighting of the neuron
            An input must be previously loaded in the neuron (see load_input)
        """
        
        self.weighting = 0.0
        for i in range(self.dimension) :    #for each dimension of the neuron
            self.weighting += self.w[i]*self.input[i]


    def calculate_activity(self, activationFunction=1) :
        """
            Calculate the activity of the neuron
            The weighting of the neuron must be previously calculated (see calculate_weighting)
        
            Parameters : 
            activationFunction -- choose the kind activation function, 0 for no function, 1 for heaviside, (0 by default)
        """
        
        if activationFunction == 0 :      #no activation function
            self.activity = self.weighting
        elif activationFunction == 1 :    #sigmoide activation function
            self.activity = sigmoide_function(self.weighting)
        else :                            #else use heaviside
            self.activity = heaviside_function(self.weighting, 0)


    def calculate_error(self, desired_output) :
        """
            Calculate the local error of the neuron
            The activity of the neuron must be previously calculated (see calculate_activity)
        
            Parameters : 
            desired_output -- the desired output for the neuron
        """
        self.error = self.activity - desired_output
        
        
    def update_weights(self, epsilon=0.1) :
        """
            Update the neuron' weights
            The error of the neuron must be previously calculated (see calculate_error)
            An input must be previously loaded in the neuron (see load_input)
        
            Parameters : 
            epsilon -- the learning rate (0.1 by default)
        """
        for i in range(self.dimension) :    #for each dimension of the neuron
            self.w[i] = self.w[i] - epsilon*self.error*self.input[i]
	

def heaviside_function(x, theta=0) :
    """
        Heaviside function 
        
        Parameters : 
        x -- the input of the function
        theta -- the threshold of the function
    """
    
    if x >= theta :     #if x is better than theta
        return 1
    else :
        return 0


def sigmoide_function(x) :
    """
        Sigmoide function 
        
        Parameters : 
        x -- the input of the function
    """
    
    return 1 / (1 + exp(-x))
