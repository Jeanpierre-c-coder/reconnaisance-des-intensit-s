# -*- coding: utf-8 -*-
"""
Utility class for Self-Adaptive Winner-takes-all (SAW) neural networks
Developed by Raphael BERGOIN
Modified by Sébastien Mick
"""
# # IMPORTS
# - Built-in
# - Third-party
from numpy import mean, std
# - Local


# # CONSTANTS


# # METHODS


# # CLASS
class SAW:
    """
    Self-Adaptive Winner-takes-all (SAW) neural network, as a single
    feed-forward layer of growing size
    """
    def __init__(self, n_features=1,
                 thresh_gamma=0.95, lr_epsilon=0.1,
                 max_neurons=100):
        """
        Set learning and growing parameters, then create empty neurons layer

        n_features: number of features (dimension) of input vectors
        thresh_gamma: tolerance threshold
        lr_epsilon: learning rate
        max_neurons: max number of neurons that can be recruited (max growth)
        """
        self.n_features = n_features
        self.gamma = thresh_gamma
        self.epsilon = lr_epsilon
        self.max_neurons = max_neurons
        self.neurons = []
        self.growth = [0]
        self.hist = [0] * max_neurons

        # Attributes that are defined later
        self.input_vec = None
        self.average = 0.
        self.standard_deviation = 0.
        self.most_active = None

    def __repr__(self):
        """
        Represent network's dimensions: current and max number of neurons as
        well as number of features
        """
        base = "<SAW network: {}/{} neurons | {} features>"
        res = base.format(len(self.neurons), self.max_neurons, self.n_features)
        return res

    def load(self, input_vec):
        """
        Load an input vector in network

        input_vec: given input vector, as a list of floats of size n_features
        """
        assert(len(input_vec) == self.n_features)
        self.input_vec = input_vec

    def calculate_nets(self):
        """
        After an input vector was loaded, calculate net value of each neuron
        """
        assert(self.input_vec is not None)
        for neur in self.neurons:
            neur.load(self.input_vec)
            neur.calculate_net()

    def calculate_avg_sd(self):
        """
        After net values were computed, get average and standard deviation of
        network output
        """
        if len(self.neurons) > 0:
            nets = [neur.net for neur in self.neurons]
            self.average = mean(nets)
            self.standard_deviation = std(nets)

    def calculate_activities(self):
        """
        After net values, average and standard deviations were computed, get
        activity of each neuron
        """
        if self.neurons:  # If layer is not empty
            nets = [neur.net for neur in self.neurons]
            imax = nets.index(max(nets))
            self.hist[imax] += 1
            self.most_active = self.neurons[imax]
            # Winner-takes-all policy
            for n in self.neurons:
                n.activity = 0.
            self.most_active.activity = self.most_active.net

    def load_and_process(self, input_vec):
        """
        Go through usual processing pipeline with given input vector. Neurons
        are NOT updated in this pipeline.
        """
        self.load(input_vec)
        self.calculate_nets()
        self.calculate_avg_sd()
        self.calculate_activities()

    def recruit_neuron(self):
        """
        After activities were computed, recruit a new neuron in layer
        """
        if len(self.neurons) < self.max_neurons:
            # De-activate previously most active neuron if it exists
            if self.neurons:
                self.most_active.activity = 0.
            # Replace it with a new neuron
            self.neurons.append(SAWNeuron(self.input_vec, act=1.))
            self.most_active = self.neurons[-1]
        else:
            self.most_active.update_weights(self.epsilon)

    def update_neurons(self):
        """
        After activities were computed, update neurons weights
        """
        if self.input_vec is not None:
            if self.neurons:  # If layer is not empty
                thresh = max(self.gamma, (self.average + self.standard_deviation))
                if self.most_active.net > thresh:
                    self.most_active.update_weights(self.epsilon)
                else:
                    self.recruit_neuron()
            else:
                self.recruit_neuron()
            self.input_vec = None
            self.growth.append(len(self.neurons))

    def reset_saw_hist(self):
        """
        Reset history of neuron activations
        """
        self.hist = [0] * self.max_neurons


class SAWNeuron:
    """
    Neuron for a Self-Adaptive Winner-takes-all (SAW) neural network
    """
    def __init__(self, weights, act=0.):
        """
        Set initial weights and corresponding number of features
        (input dimension)

        weights: initial weights
        """
        self.n_features = len(weights)
        self.weights = weights
        self.activity = act

        # Attributes that wil be defined later
        self.input_vec = None
        self.net = 0.

    def load(self, input_vec):
        """
        Load input vector in neuron

        input_vec: input vector, as a list of n_features floats
        """
        assert(len(input_vec) == self.n_features)
        self.input_vec = input_vec

    def calculate_net(self):
        """
        After an input vector was loaded, calculate net value (output)
        """
        assert(self.input_vec is not None)
        total = sum((abs(weight - val) for weight, val in zip(self.weights,
                                                              self.input_vec)))

        self.net = 1 - total / self.n_features

    def update_weights(self, epsilon=0.1):
        """
        After activity was computed, update neuron weights

        epsilon: learning rate
        """
        factor = epsilon * (1 - self.activity)
        self.weights = [weight + factor * (val - weight) for val, weight
                        in zip(self.input_vec, self.weights)]
