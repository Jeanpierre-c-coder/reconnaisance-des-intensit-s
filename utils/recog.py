# -*- coding: utf-8 -*-
"""
Classes for managing multiple neuron layers of various types, as a single model
for Facial Expression Recognition (FER)
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from abc import ABC, abstractmethod
from itertools import chain, cycle
# - Third-party
import numpy as np
# - Local
from .lms import LMS
from .saw import SAW
from .stm import STM


# # CONSTANTS
# Emotional states and facial expressions
PRIMARY_STATES = ["neutral", "happy", "sad", "surprised", "angry"]
N_STATES = len(PRIMARY_STATES)
TRAIT_GROUPS = ["eyebrows", "lip_corners", "mouth_opening"]
TRAIT_LEVELS = ["low", "neutral", "high"]
N_TRAIT_LEVELS = [3, 3, 3]
N_GROUPS = len(N_TRAIT_LEVELS)
# Layer sizes
GROUP_SIZE = sum(N_TRAIT_LEVELS)
SAW_MAX_SIZE = 250
N_KEYPOINTS = 15
N_FEATURES = 81
# Learning parameters
VIGILANCE_TH = 0.96
SAW_LEARNING_RATE = 0.01
LMS_LEARNING_RATE = 0.01
LMS_MIN, LMS_MAX = 0.1, 0.11
STM_LOCAL_ALPHA = 1. / N_KEYPOINTS
STM_LOCAL_BETA = 1. - STM_LOCAL_ALPHA
STM_PRED_ALPHA, STM_PRED_BETA = 0.6, 0.3
# Reference for WTA-based expression of emotional states and facial traits
REF_STATES = np.eye(N_STATES).tolist()
# Each element represents a complete facial expression (where all trait
# groups are recruited) as a set of "activation" levels
REF_TRAIT_LEVELS = [[[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]],  # neutral
                    [[0., 1., 0.], [0., 0., 1.], [0., 1., 0.]],  # happy
                    [[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]],  # sad
                    [[0., 0., 1.], [0., 1., 0.], [0., 0., 1.]],  # surprised
                    [[1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]]  # angry
REF_TRAIT_3D = np.array([[1, 1, 1],
                         [1, 2, 1],
                         [1, 0, 1],
                         [2, 1, 2],
                         [0, 1, 0]])


# # METHODS
def unpack_prim(emo):
    """
    Unpack activation levels and information strings corresponding to given
    primary emotion
    """
    levels = REF_TRAIT_LEVELS[emo]
    lev_str = "".join((str(it) for it in project_3d(levels)))

    return emo, levels, PRIMARY_STATES[emo], lev_str


def unpack_seco(state, exp_mode):
    """
    Unpack activation levels and information strings corresponding to given
    internal state, with respect to secondary emotions
    """
    levels, emo = state[:2]
    lev_str = "".join((str(it) for it in project_3d(levels)))
    emo_str = PRIMARY_STATES[emo] if type(emo) is int else str(emo)

    group = state[2] if exp_mode > 1 else None

    return emo, levels, emo_str, lev_str, group


def get_highest_group(lms_layers):
    """
    Find group of traits corresponding to highest non-neutral activity
    """
    lms_levels = [[neur.activity for neur in layer.neurons]
                  for layer in lms_layers]
    non_neutral_mask = project_3d(lms_levels) != 1
    if np.any(non_neutral_mask):
        nn_groups = np.where(non_neutral_mask)[0]
        nn_acts = [max(lms_levels[g]) for g in nn_groups]
        res = nn_acts.index(max(nn_acts))
    else:
        res = -1

    return res


def get_likely_emo_state(levels, details=False):
    """
    Get most likely emotional state based on elementary expressions
    """
    lev_3d = project_3d(levels)
    # Is any trait group associated with a non-neutral activity?
    non_neutral_mask = lev_3d != 1
    if np.any(non_neutral_mask):
        # Indices of all non-neutral groups
        nn_groups = np.where(non_neutral_mask)[0]
        # Find primary state corresponding to each non-neutral group
        nn_acts = [max(levels[g]) for g in nn_groups]
        nn_states = []
        for group in nn_groups:
            corr_nn_mask = REF_TRAIT_3D[:, group] == lev_3d[group]
            nn_states.append(np.where(corr_nn_mask)[0][0])
        # Count how many times each primary state appears
        unique_states, unique_counts = np.unique(nn_states, return_counts=True)
        # Compute cumulative activity levels from each non-neutral group
        act_dict = {st: 0. for st in unique_states}
        for st, act in zip(nn_states, nn_acts):
            act_dict[st] += act
        cum_acts = sorted(act_dict.items(), key=lambda item: item[1],
                          reverse=True)
    else:
        cum_acts = [(0, 3.)]
    res = cum_acts if details else cum_acts[0][0]

    return res


def get_emo_state_proj(levels):
    """
    Get "closest" emotional state based on Euclidian distance when projecting
    activation levels in a 3D space
    """
    levels_3d = project_3d(levels)
    dist = np.linalg.norm(REF_TRAIT_3D - levels_3d, axis=1, ord=2)
    state = np.argmin(dist)

    return state


def reshape_as_levels(vec):
    """
    Reshape a 1-D vector (list of floats) as a set of activation levels, in the
    form of a list of lists of floats with appropriate sizes
    """
    assert(len(vec) == GROUP_SIZE)

    res = []
    start_ind = 0
    for n_lev in N_TRAIT_LEVELS:
        end_ind = start_ind + n_lev
        res.append(vec[start_ind:end_ind])
        start_ind = end_ind

    return res


def project_3d(levels):
    """
    Project a set of activation levels onto an abstract 3-D space
    """
    res = []
    for group in levels:
        res.append(np.argmax(group))

    return np.array(res)


def get_levels_from_3d(vec):
    """
    Get a set of activation levels from given vector in abstract 3-D space
    """
    lev = []
    for val, n_lev in zip(vec.astype(int), N_TRAIT_LEVELS):
        grp_lev = [0.] * n_lev
        grp_lev[val] = 1.0
        lev.append(grp_lev)

    return lev


def fit_vec(vec, size):
    """
    Fit given vector to given size:
    - if too many elements, extra ones are discarded
    - if too few, vector is padded with zeros

    vec: list of floats
    size: size of output
    """
    n_diff = size - len(vec)
    res = vec[:size] if n_diff <= 0 else vec + [0.] * n_diff

    return res


def chain_output_vec(layers):
    """
    Chain activities of neurons from multiple layers as one output vector
    """
    res = []
    for layer in layers:
        res.extend([neur.activity for neur in layer.neurons])

    return res


# # CLASSES
class Model(ABC):
    """
    Generic interface for a FER model
    """

    @abstractmethod
    def feed(self, input_vec, curr_state=None):
        """
        Feed given input_samples into successive neuron layers
        """
        pass

    @abstractmethod
    def clear_stm(self):
        """
        Reset activities of local STM layer
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update weights and activities of neurons based on error computed from
        input_samples
        """
        pass

    @abstractmethod
    def predict(self, prod):
        """
        Make an FER prediction based on the model’s outputs
        """
        pass


class ModelByState(Model):
    """
    FER model where an internal state corresponds to a primary emotion
    """

    def __init__(self, saw_max=SAW_MAX_SIZE):
        """
        Layers are arranged as follows:
        - SAW layer processing high-dimensional input from image processing
        - LMS layer learning a relation between SAW output and internal state
        - "Local" STM layer filtering LMS output over a batch
        - "Prediction" STM layer filtering local STM output over time
        """
        self.saw = SAW(N_FEATURES, VIGILANCE_TH,
                       SAW_LEARNING_RATE, saw_max)
        self.lms = LMS(saw_max, N_STATES, LMS_LEARNING_RATE,
                       LMS_MIN, LMS_MAX)
        self.local_stm = STM(N_STATES, STM_LOCAL_ALPHA, STM_LOCAL_BETA)
        self.pred_stm = STM(N_STATES, STM_PRED_ALPHA, STM_PRED_BETA)

    def __repr__(self):
        """
        Represent model’s structure as input and output sizes
        """
        res = "<ModelByState | Dimensions {} in {} out>"

        return res.format(self.saw.n_features,
                          self.pred_stm.n_neurons)

    def feed(self, input_vec, curr_state=None):
        """
        Parameters
        - input_vec: list of floats of size N_FEATURES
        - curr_state: index of current internal state, as a float
        """
        # Distinguish learning (when robot internal state is known)
        # and prediction
        state = None if curr_state is None else REF_STATES[curr_state]

        self.saw.load_and_process(input_vec)
        saw_output = fit_vec([neur.activity for neur in self.saw.neurons],
                             self.saw.max_neurons)
        self.lms.load_and_process(saw_output, state)
        if state is None:  # Feed STM layer only during prediction
            self.local_stm.integrate([n.activity for n in self.lms.neurons])

    def clear_stm(self):
        self.local_stm.clear()

    def update(self):
        self.saw.update_neurons()
        self.lms.update_neurons()

    def predict(self, output="local"):
        # Update prediction STM layers
        self.pred_stm.integrate(self.local_stm.neurons)
        if output == "prod":
            # Multiply with LMS output to get final output vector (??)
            pred = np.multiply([n.activity for n in self.lms.neurons],
                               self.pred_stm.neurons)
        elif output == "pred":
            pred = self.pred_stm.neurons
        else:
            pred = self.local_stm.neurons

        return pred, np.argmax(pred)


class ModelByGroup(Model):
    """
    FER model where an internal state is represented as a set of activation
    levels for different groups of facial traits
    """

    def __init__(self, saw_max=SAW_MAX_SIZE):
        """
        Layers are arranged as follows:
        - SAW layer to process high-dimensional input
        - One LMS layer for each trait group
        - One local STM layer "concatenating" outputs from all LMS layers
          (because neurons are independent in an STM layer)
        - One prediction STM layer
        """
        self.saw = SAW(N_FEATURES, VIGILANCE_TH,
                       SAW_LEARNING_RATE, saw_max)
        # One dedicated LMS layer for each group of facial traits
        self.lms_layers = [LMS(saw_max, n_lev,
                               LMS_LEARNING_RATE, LMS_MIN, LMS_MAX)
                           for n_lev in N_TRAIT_LEVELS]
        # Given that each STM neuron is independent of the others, a single
        # layer can be used for all groups of facial traits
        self.local_stm = STM(GROUP_SIZE, STM_LOCAL_ALPHA, STM_LOCAL_BETA)
        self.pred_stm = STM(GROUP_SIZE, STM_PRED_ALPHA, STM_PRED_BETA)

    def __repr__(self):
        """
        Represent model’s structure as input size, output size and number of
        LMS layers
        """
        res = "<ModelByGroup with {} LMS layers | Dimensions {} in {} out>"

        return res.format(len(self.lms_layers),
                          self.saw.n_features,
                          self.pred_stm.n_neurons)

    def feed(self, input_vec, curr_state=None):
        """
        Parameters
        - input_vec: list of floats of size N_FEATURES
        - curr_state: current internal state, as a list of lists of floats
        """
        self.saw.load_and_process(input_vec)
        saw_output = fit_vec([neur.activity for neur in self.saw.neurons],
                             self.saw.max_neurons)

        # Define which LMS layers will learn
        if curr_state is None:  # Prediction phase
            # LMS layers do not learn but are still fed by SAW layer
            zip_learn = zip(self.lms_layers, cycle((None, )))
        else:
            state_3d = project_3d(curr_state)
            if np.any(state_3d == 1):  # At least one non-neutral level
                # Only learn where non-neutral, skip feeding other layers
                zip_learn = [(lms, act) for lms, act, lev in
                             zip(self.lms_layers, curr_state, state_3d)
                             if lev != 1]
            else:  # Fully neutral face: all layers learn
                zip_learn = zip(self.lms_layers, curr_state)
        for lms, act in zip_learn:
            lms.load_and_process(saw_output, act)

        if curr_state is None:  # Feed STM layer only during prediction
            self.local_stm.integrate(chain_output_vec(self.lms_layers))

    def feed_select(self, input_vec, curr_state, learn_group):
        """
        Parameters
        - input_vec: list of floats of size N_FEATURES
        - curr_state: current internal state, as a list of lists of floats
        - learn_group: index of the group currently learning
        """
        self.saw.load_and_process(input_vec)
        saw_output = fit_vec([neur.activity for neur in self.saw.neurons],
                             self.saw.max_neurons)
        self.lms_layers[learn_group].load_and_process(saw_output,
                                                      curr_state[learn_group])

    def clear_stm(self):
        self.local_stm.clear()

    def update(self):
        self.saw.update_neurons()
        for lms in self.lms_layers:
            lms.update_neurons()

    def predict(self, comp=False, output="local"):
        self.pred_stm.integrate(self.local_stm.neurons)
        if output == "prod":
            lms_output = chain_output_vec(self.lms_layers)
            pred = np.multiply(lms_output, self.pred_stm.neurons)
        elif output == "pred":
            pred = self.pred_stm.neurons
        else:
            pred = self.local_stm.neurons
        state = reshape_as_levels(pred)
        state_3d = project_3d(state)

        if comp:
            # If both mouth groups have non-neutral levels, make them compete
            if np.all(state_3d[1:] != 1):
                # Find "losing" group and inhibit its activity
                mouth_max_levs = [state[g][state_3d[g]] for g in (1, 2)]
                min_max_lev = min(mouth_max_levs)
                min_g = mouth_max_levs.index(min_max_lev) + 1
                state[min_g][1] = min_max_lev
                state[min_g][state_3d[min_g]] = min_max_lev * 0.9
                pred = list(chain(*state))  # Flatten levels

        res = get_likely_emo_state(state)

        return pred, res


class ModelThreeway(Model):
    """
    FER model with three distinct pathways selectively learning to recognize
    activation levels of facial trait groups
    """

    def __init__(self, saw_max=SAW_MAX_SIZE):
        """
        Each pathway contains
        - One SAW layer
        - One LMS layer
        The three pathways are merged with
        - One local STM layer "concatenating" outputs from all LMS layers
          (because neurons are independent in an STM layer)
        - One prediction STM layer
        """
        self.ways = [(SAW(N_FEATURES, VIGILANCE_TH, SAW_LEARNING_RATE, saw_max),
                      LMS(saw_max, n_lev, LMS_LEARNING_RATE, LMS_MIN, LMS_MAX))
                     for n_lev in N_TRAIT_LEVELS]
        self.local_stm = STM(GROUP_SIZE, STM_LOCAL_ALPHA, STM_LOCAL_BETA)
        self.pred_stm = STM(GROUP_SIZE, STM_PRED_ALPHA, STM_PRED_BETA)

    def __repr__(self):
        """
        Represent model’s structure as input size, output size and number of
        LMS layers
        """
        res = "<ModelThreeway | Dimensions {} in {} out>"

        return res.format(self.ways[0][0].n_features,
                          self.pred_stm.n_neurons)

    def feed(self, input_vec, curr_state=None):
        """
        Parameters
        - input_vec: list of floats of size N_FEATURES
        - curr_state: current internal state expressing activation levels
        """
        state = cycle((None, )) if curr_state is None else curr_state

        for (saw, lms), act in zip(self.ways, state):
            saw.load_and_process(input_vec)
            saw_output = fit_vec([n.activity for n in saw.neurons],
                                 saw.max_neurons)
            lms.load_and_process(saw_output, act)

        if curr_state is None:  # Feed STM layer only during prediction
            lms_layers = [way[1] for way in self.ways]
            self.local_stm.integrate(chain_output_vec(lms_layers))

    def feed_select(self, input_vec, curr_state, learn_group):
        """
        Parameters
        - input_vec: list of floats of size N_FEATURES
        - curr_state: current internal state, as a list of lists of floats
        - learn_group: index of the group currently learning
        """
        # SAW and LMS layers from selected group
        saw, lms = self.ways[learn_group]
        saw.load_and_process(input_vec)
        saw_output = fit_vec([n.activity for n in saw.neurons], saw.max_neurons)
        lms.load_and_process(saw_output, curr_state[learn_group])

    def clear_stm(self):
        self.local_stm.clear()

    def update(self):
        for saw, lms in self.ways:
            saw.update_neurons()
            lms.update_neurons()

    def predict(self, comp=False, output="local"):
        if output == "pred":
            self.pred_stm.integrate(self.local_stm.neurons)
            pred = self.pred_stm.neurons
        else:
            pred = self.local_stm.neurons
        state = reshape_as_levels(pred)
        state_3d = project_3d(state)

        if comp:
            # If both mouth groups have non-neutral levels, make them compete
            if np.all(state_3d[1:] != 1):
                # Find "losing" group and inhibit its activity
                mouth_max_levs = [state[g][state_3d[g]] for g in (1, 2)]
                min_max_lev = min(mouth_max_levs)
                min_g = mouth_max_levs.index(min_max_lev) + 1
                state[min_g][1] = min_max_lev
                state[min_g][state_3d[min_g]] = min_max_lev * 0.9
                pred = list(chain(*state))  # Flatten levels

        res = get_likely_emo_state(state)

        return pred, res
