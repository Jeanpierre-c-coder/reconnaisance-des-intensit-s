# -*- coding: utf-8 -*-
"""
Utility class for Least Mean Square (LMS) neural networks
Developped by Raphael BERGOIN
Modified by Sébastien Mick
"""
# # IMPORTS
# - Built-in
from math import exp
from random import uniform
# - Third-party
from numpy import argmax
# - Local


# # CONSTANTS


# # METHODS


# # CLASS
class LMS:
    """
    Least Mean Square (LMS) neural network, as a single feed-forward layer
    """
    def __init__(self, n_features=1, n_neurons=1,
                 epsilon=0.1, range_min=0., range_max=1., acc=False):
        """
        Set learning parameters and create layer of neurons

        n_features: number of features (dimension) of input vectors
        n_neurons: number of neurons in network
        epsilon: learning rate
        range_min: minimum initial value for a neuron weight
        range_max: maximum initial value for a neuron weight
        """
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.epsilon = epsilon
        self.range_min = range_min
        self.range_max = range_max

        neuron = LMSNeuronAcc if acc else LMSNeuron
        self.neurons = [neuron(n_features, range_min, range_max)
                        for _ in range(n_neurons)]

    def __repr__(self):
        """
        Represent network's current state as a list of all neurons activities
        """
        acts = [round(neur.activity, 3) for neur in self.neurons]
        res = "<LMS network {}>".format(acts)
        return res

    def calculate_weightings(self, input_vec):
        """
        Load an input vector in network and calculate weighting of each neuron

        input_vec: given input vector, as a list of size n_features
        """
        assert(len(input_vec) == self.n_features)
        for neur in self.neurons:
            neur.load(input_vec)
            neur.calculate_weighting()

    def calculate_activities(self):
        """
        After weightings were computed, calculate activity of each neuron
        """
        for neur in self.neurons:
            neur.calculate_activity()

    def calculate_errors(self, goal_output):
        """
        After activities were computed, calculate local error of each neuron

        goal_output: desired output vector, as a list of size n_neurons
        """
        assert(len(goal_output) == self.n_neurons)

        for neur, goal_val in zip(self.neurons, goal_output):
            neur.calculate_error(goal_val)

    def load_and_process(self, input_vec, output_vec):
        """
        Go through usual processing pipeline with given input and output
        vectors. Neurons are NOT updated in this pipeline.
        """
        self.calculate_weightings(input_vec)
        self.calculate_activities()
        if output_vec is not None:
            self.calculate_errors(output_vec)

    def update_neurons(self):
        """
        After errors were computed, update neurons weights
        """
        for neur in self.neurons:
            neur.update_weights(self.epsilon)
            neur.input_vec = None  # Reset input memory after updating


class LMSNeuron:
    """
    Neuron for Least Mean Square (LMS) neural network
    """
    def __init__(self, n_features=1, range_min=0., range_max=1.):
        """
        Set number of features (dimension) and range of possible values for
        initial weights

        n_features: dimension of input vectors
        range_min: minimum initial value for a weight
        range_max: maximum initial value for a weight
        """
        self.n_features = n_features
        self.weights = [uniform(range_min, range_max)
                        for _ in range(n_features)]

        # Attributes that will be defined later
        self.input_vec = None
        self.weighting = 0.
        self.error = 0.
        self.activity = 0.

    def load(self, input_vec):
        """
        Load input vector in neuron

        input_vec: input vector, as a list of n_features floats
        """
        self.input_vec = input_vec

    def calculate_weighting(self):
        """
        After an input vector was loaded, calculate weighting
        """
        self.weighting = sum((w * inp for w, inp in zip(self.weights,
                                                        self.input_vec)))

    def calculate_activity(self, act_func=""):
        """
        After weighting was computed, calculate neuron activity

        act_func: activation function {"sig"|"heav"}
        """
        if act_func == "sig":  # Sigmoid
            self.activity = 1. / (1 + exp(-self.weighting))
        elif act_func == "heav":  # Heaviside
            self.activity = 0. if self.weighting < 0 else 1.
        else:  # Identity as activation function
            self.activity = self.weighting

    def calculate_error(self, goal_output):
        """
        After activity was computed, calculate neuron error on output

        goal_output: desired output value
        """
        self.error = self.activity - goal_output

    def update_weights(self, epsilon=0.1):
        """
        After error was computed, update neuron weights

        epsilon: learning rate
        """
        if self.input_vec is not None:
            self.weights = [w - epsilon * self.error * inp
                            for w, inp in zip(self.weights, self.input_vec)]


class LMSNeuronAcc:
    """
    Neuron for Least Mean Square (LMS) neural network that updates faster
    provided its inputs are winner-takes-all vectors
    """
    def __init__(self, n_features=1, range_min=0., range_max=1.):
        """
        Set number of features (dimension) and range of possible values for
        initial weights

        n_features: dimension of input vectors
        range_min: minimum initial value for a weight
        range_max: maximum initial value for a weight
        """
        self.n_features = n_features
        self.weights = [uniform(range_min, range_max)
                        for _ in range(n_features)]

        # Attributes that will be defined later
        self.max_ind = None
        self.weighting = 0.
        self.error = 0.
        self.activity = 0.

    def load(self, input_vec):
        """
        Load input vector in neuron

        input_vec: winner-takes-all input vector
        """
        self.max_ind = argmax(input_vec)

    def calculate_weighting(self):
        """
        After an input vector was loaded, calculate weighting
        """
        self.weighting = self.weights[self.max_ind]

    def calculate_activity(self):
        """
        After weighting was computed, calculate neuron activity

        act_func: activation function {"sig"|"heav"}
        """
        self.activity = self.weighting

    def calculate_error(self, goal_output):
        """
        After activity was computed, calculate neuron error on output

        goal_output: desired output value
        """
        self.error = self.activity - goal_output

    def update_weights(self, epsilon=0.1):
        """
        After error was computed, update neuron weights

        epsilon: learning rate
        """
        if self.max_ind is not None:
            new_w = self.weights.copy()
            new_w[self.max_ind] -= epsilon * self.error
            self.weights = new_w
