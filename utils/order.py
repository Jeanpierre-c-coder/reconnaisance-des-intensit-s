# -*- coding: utf-8 -*-
"""
Toolbox for generating trial orders
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from random import sample, shuffle
from itertools import combinations, product
from pickle import dump
# - Third-party
import numpy as np
# - Local
from . import recog as rec


# # CONSTANTS
ELEM_TRAITS = [([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.]], 0),  # Neutral
               ([[1., 0., 0.], [0., 1., 0.], [0., 1., 0.]], 4),  # EB:Angry
               ([[0., 0., 1.], [0., 1., 0.], [0., 1., 0.]], 3),  # EB:Surprised
               ([[0., 1., 0.], [1., 0., 0.], [0., 1., 0.]], 2),  # LC:Sad
               ([[0., 1., 0.], [0., 0., 1.], [0., 1., 0.]], 1),  # LC:Happy
               ([[0., 1., 0.], [0., 1., 0.], [1., 0., 0.]], 4),  # MO:Angry
               ([[0., 1., 0.], [0., 1., 0.], [0., 0., 1.]], 3)]  # MO:Surprised
# Which group is activated by which elementary expression
ELEM_GROUP_MASK = {0: [1, 2],  # Eyebrows
                   1: [3, 4],  # Lip corners
                   2: [5, 6]}  # Mouth opening
# How many times the neutral face appears in a block
N_REPEATS_NEUTRAL = 3
# Fixed sequence of elementary expressions
FIXED_ELEM_SEQ = ((1, 0, 2),  # Eyebrows
                  (3, 0, 4),  # Lip corners
                  (5, 0, 6))  # Mouth opening
# Fixed sequence of mixed expressions, not including those mixing mouth groups
FIXED_MIX_SEQ = ((1, 3), (1, 4), (1, 6),
                 (2, 3), (2, 4), (2, 5))


# # METHODS
def add_elem_traits(t1, t2):
    """
    Mix two elementary traits by "adding" the corresponding activation levels
    """
    # Unpack parameters
    lev1, emo1 = t1
    lev2, emo2 = t2
    # Project onto 3D space and add as vectors
    sum_vec = rec.project_3d(lev1) + rec.project_3d(lev2)
    # Substract one to every coordinate because 1 = neutral
    sum_lev = rec.get_levels_from_3d(sum_vec - np.ones(rec.N_GROUPS))

    return sum_lev, (emo1, emo2)


def get_order_block(n_blocks, n_elems):
    """
    Generate a block-randomized order
    """
    next_block = sample(range(n_elems), n_elems)  # First block
    order = next_block
    for _ in range(n_blocks - 1):
        block_ok = False
        while not block_ok:
            next_block = sample(range(n_elems), n_elems)
            # Ensure that no duplicates appear in consecutive elements
            block_ok = order[-1] != next_block[0]
        order.extend(next_block)

    return order


def get_learning_order(n_blocks, seq, return_inds=True):
    """
    Generate a trial order for learning phase, with elements drawn from
    different sets depending on the experimental mode
    """
    order = get_order_block(n_blocks, len(seq))
    res = order if return_inds else [seq[i] for i in order]

    return res


def get_order_mix(n_trials=0):
    """
    Generate an order of mixed facial expressions, involving two different trait
    groups associated with two different primary emotions
    """
    mix_list = []
    # Go through all combinations of two trait groups out of three
    for g1, g2 in combinations(range(rec.N_GROUPS), 2):
        prod_inds = product(ELEM_GROUP_MASK[g1], ELEM_GROUP_MASK[g2])
        prod = [(ELEM_TRAITS[i], ELEM_TRAITS[j]) for i, j in prod_inds]
        # Only mix traits associated with different emotions
        mix = [add_elem_traits(t1, t2) for t1, t2 in prod if t1[1] != t2[1]]
        mix_list.extend(mix)
    shuffle(mix_list)
    order = mix_list[:n_trials] if n_trials > 0 else mix_list

    return order


def get_order_mix_fixed_seq(n_trials=0):
    """
    """
    mix_list = []
    for e1, e2 in FIXED_MIX_SEQ:
        mix_list.append(add_elem_traits(ELEM_TRAITS[e1], ELEM_TRAITS[e2]))
    order = mix_list[n_trials] if n_trials > 0 else mix_list

    return order


def get_order_elem():
    """
    Generate a block of elementary facial expressions, sorted by trait group and
    activation level
    """
    block = []
    for ind, series in enumerate(FIXED_ELEM_SEQ):
        block.extend([(*ELEM_TRAITS[i], ind) for i in series])

    return block


def save_order(path, order):
    """
    Serialize a sequence object corresponding to an order of trials
    """
    with open(path, 'wb') as ofile:
        dump(order, ofile)


# # MAIN
if __name__ == "__main__":
    order_by_state = get_learning_order(5, range(rec.N_STATES), True)
    save_order("utils/order_prim.pkl", order_by_state)

    # Combine elementary expressions with "full" expressions
    # Each non-neutral trait should appear twice
    full_traits = [(rec.REF_TRAIT_LEVELS[emo], emo)
                   for emo in range(1, rec.N_STATES)]
    elem_and_full_traits = ELEM_TRAITS[1:] + full_traits
    elem_and_full_traits.extend([ELEM_TRAITS[0]] * N_REPEATS_NEUTRAL)
    # order_by_group = get_learning_order(4, elem_and_full_traits, False)
    # save_order("utils/order_.pkl", order_by_group)

    order_by_group = get_order_mix_fixed_seq()
    save_order("utils/order_mix.pkl", order_by_group)

    order_by_group = get_order_elem() * 4
    save_order("utils/order_elem.pkl", order_by_group)
