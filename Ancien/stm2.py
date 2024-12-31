#!/usr/bin/env python

"""
Contains the classes and the functions needed in order to create a Short Term Memory(STM) neural network

Developped by Raphael BERGOIN
Modified by Dorian IBERT adding coef for sliding
"""

from random import *
from math import *
import string
import numpy as np
import codecs
import random


class STM :
    """
        Class to represent a Short Term Memory(STM) neural network
    """

    def __init__(self, nb_neurons=1, norm=1.0, coef=1.0) :
        """
            Constructor of the class LMS
        
            Parameters : 
            nb_neurons -- the dimension of the input/output
            norm -- the norm value to multiply the input
            coef -- the oblivion factor
        """
        
        self.nb_neurons = nb_neurons
        self.norm = norm
        self.neurons = [0.0*norm for i in range(nb_neurons)]
        self.coef = coef
        
    
    def integrate(self, data) :
        """
            Integrate/accumulate a new data to the network
        
            Parameters : 
            data -- the data input vector given to the neurons
        """
        
        for i in range(self.nb_neurons) :    #for each neurons of the network
            self.neurons[i]+=self.norm*data[i]
            
    def slide(self,data):
        """
            Allow sliding effect
        """
        if self.neurons == [0.0 for i in range(self.nb_neurons)]:
            for i in range(self.nb_neurons) :    #for each neurons of the network
                self.neurons[i]+=self.norm*data[i]
        else:
            for i in range(self.nb_neurons): #for each neurons of the network
                self.neurons[i] = (self.neurons[i]*self.coef + data[i])/2
            
    def setNorm(self, norm) :
        """
            Set the value of the norm of the network
        
            Parameters : 
            norm -- the norm value to multiply the input
        """
            
        self.norm = norm
        
    
    def clearNetwork(self) :
        """
            Put neurons' value of the network to 0
        """
    
        self.neurons = [0.0*self.norm for i in range(self.nb_neurons)]
    
    
    def display_output(self) :
        """
            Display the output of each neuron of the network
        """
        
        for i in range(self.nb_neurons) :   #for each neurons of the network
            print(self.neurons[i])
        print("\n")
           