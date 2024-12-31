#!/usr/bin/env python

"""
Contains the classes and the functions needed in order to create a Self Adaptive Winner takes all(SAW) neural network

Developped by Raphael BERGOIN
"""

from random import *
from math import *
import string
import numpy as np
import codecs
import random


class SAW :
    """
        Class to represent a Self Adaptive Winner takes all(SAW) neural network
    """


    def __init__(self, dimension=1, gamma=0.95, nb_neurons_max = 100, epsilon=0.1) :
        """
            Constructor of the class SAW
        
            Parameters : 
            dimension -- the dimension of the neurons (1 by default)
            gamma -- the tolerance threshold (0.95 by default)
            nb_neurons_max -- the maximum number of neurons that can be recruited (100 by default)
            epsilon -- the learning rate (0.1 by default)
        """
        
        self.nb_neurons = 0
        self.nb_neurons_max = nb_neurons_max
        self.dimension = dimension
        self.gamma = gamma
        self.epsilon = epsilon
        self.neurons = []
        self.maximums = [0.0 for x in range(dimension)]             #for normalization of input (not necessary)
        self.minimums = [1000000.0 for x in range(dimension)]       #for normalization of input (not necessary)


    def load_input(self, data) :
        """
            Load an input in the network
        
            Parameters : 
            data -- the data input vector given to the network
        """
        
        self.input = data


    def calculate_average(self) :
        """
            Calculate the average of the output of the network
            An input must be previously loaded in the network (see load_input)
        """
        
        total = 0.0
        for i in range(self.nb_neurons) :   #for each neurons of the network
            total += self.neurons[i].net
        if self.nb_neurons != 0 :           #if at least one network has been recruited
            self.average = total/self.nb_neurons

    def calculate_standard_deviation(self) :
        """
            Calculate the standard deviation of the output of the network
            An input must be previously loaded in the network (see load_input)
        """
        
        total = 0.0
        for i in range(self.nb_neurons) :   #for each neurons of the network
            total += (self.neurons[i].net-self.average)**2
        if self.nb_neurons != 0 :           #if at least one network has been recruited
            self.standard_deviation = sqrt(total/self.nb_neurons)


    def calculate_nets(self) :
        """
            Calculate the output of each neuron of the network
            An input must be previously loaded in the network (see load_input)
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            self.neurons[i].load_input(self.input) 
            self.neurons[i].calculate_net()


    def calculate_activities(self) :
        """
            Calculate the activity of each neuron of the network
            The output of each neuron of the network must be previously calculated (see calculate_nets)
            The average of the network must be previously calculated (see calculate_average)
            The standard deviation of the network must be previously calculated (see calculate_standard_deviation)
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            self.neurons[i].calculate_activity(self.average, self.standard_deviation, self.gamma)


    def get_most_active_neuron(self) :
        """
            Find the most active neuron in the network and store its index
            The output of each neuron of the network must be previously calculated (see calculate_nets)
        """
        
        activity_max = 0.0
        self.k = 0
        for i in range(self.nb_neurons) :   #for each neurons of the network
            if self.neurons[i].net >= self.neurons[self.k].net :    #if the output of the neuron i is better than that of neuron k
                activity_max = self.neurons[i].net
                self.k = i
    
    
    def recruit_neuron(self) :
        """
            Recruit a new neuron in the network
            An input must be previously loaded in the network (see load_input)
        """
        
        if self.nb_neurons < self.nb_neurons_max :  #if the network is not full
            self.neurons.append(Neuron(self.input, self.dimension))     #create and add a new neuron with the information of the input
            self.nb_neurons += 1
        else :     #else we update anyway
            self.neurons[self.k].update_weights(self.epsilon)   #we update the weights


    def update_neuron(self) :
        """
            Update the neurons' weights of the network
            The output of each neuron of the network must be previously calculated (see calculate_nets)
            The activity of each neuron of the network must be previously calculated (see calculate_activities)
        """
        
        if self.nb_neurons == 0 :   #if the network is empty
            self.recruit_neuron()
        else :
            self.get_most_active_neuron()
            if self.neurons[self.k].net > self.gamma :  #if the most active neuron has an output below the vigilance threshold
                self.neurons[self.k].update_weights(self.epsilon)   #we update the weights
            else :                                      #else we recruit a new neuron
                self.recruit_neuron()
            


class Neuron :
    """
        Class to represent a simple neuron for a Self Adaptive Winner takes all(SAW) neural network
    """

    def __init__(self, data, dimension=1) :
        """
            Constructor of the class Neuron
        
            Parameters : 
            dimension -- the dimension of neurons (1 by default)
            data -- intial weights of the neuron
        """
        
        self.dimension = dimension
        self.w = data


    def load_input(self, data) :
        """
            Load an input in the neuron
        
            Parameters : 
            data -- the data input vector given to the neuron
        """
        
        self.input = data


    def calculate_net(self) :
        """
            Calculate the output of the neuron
            An input must be previously loaded in the neuron (see load_input)
        """
        
        total = 0
        for i in range(self.dimension) :    #for each dimension of the neuron
            total += abs(self.w[i]-self.input[i])
        self.net = 1 - (total/self.dimension)


    def calculate_activity(self, average, sigma, gamma=0.95) :
        """
            Calculate the activity of the neuron
            The output of the neuron must be previously calculated (see calculate_net)
            
            Parameters : 
            average -- the average of the output of the network
            sigma -- the standard deviation of the output of the network
            gamma -- the tolerance threshold (0.95 by default)
        """
        self.activity = self.net*heaviside_function(self.net, max(gamma, (average+sigma)))


    def update_weights(self, epsilon=0.1) :
        """
            Update the neurons' weights
            The activity of the neuron must be previously calculated (see calculate_activity)
            An input must be previously loaded in the neuron (see load_input)
            
            Parameters : 
            epsilon -- the learning rate (0.1 by default)
        """
        
        for i in range(self.dimension) :    #for each dimension of the neuron
            self.w[i] += epsilon*(self.input[i]-self.w[i])*(1-self.activity)        



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



