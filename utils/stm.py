# -*- coding: utf-8 -*-
"""
Utility class for Short-Term Memory(STM) neural networks
Developed by Raphael BERGOIN
Modified by Dorian IBERT (adding coef for sliding) and Sébastien Mick
"""
# # IMPORTS
# - Built-in
# - Third-party
# - Local


# # CONSTANTS


# # CLASS
class STM:
    """
    Short-Term Memory (STM) neural network, as a single recurrent layer
    """
    def __init__(self, n_neurons=1, input_alpha=1.0, recur_beta=1.0):
        """
        Create neurons layer and set weighting factors

        n_neurons: layer size, corresponding to input/output dimension
        input_alpha: factor applied to input
        recur_beta: factor applied to recurrent link (oblivion factor)
        """
        self.alpha = input_alpha
        self.beta = recur_beta
        self.n_neurons = n_neurons
        self.neurons = [0.] * n_neurons

    def __repr__(self):
        """
        Represent network's current state as a list of all neurons activities
        """
        acts = [round(neur, 3) for neur in self.neurons]
        res = "<STM network {}>".format(acts)
        return res

    def clear(self):
        """
        Clear layer by resetting all neuron activities to 0
        """
        self.neurons = [0.] * self.n_neurons

    def integrate(self, input_vec):
        """
        Integrate data sample given as input vector

        input: data sample, as a list of floats of same size as layer
        """
        self.neurons = [self.alpha * input_i + self.beta * neuron_i
                        for input_i, neuron_i in zip(input_vec, self.neurons)]
