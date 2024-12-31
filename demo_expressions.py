# -*- coding: utf-8 -*-
"""
Run a demonstration of various types of robot facial expressions employed during
FER experiment
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from sys import argv
# - Third-party
# - Local
import utils.robot as rob
import utils.recog as rec
import utils.order as orut


# # CONSTANTS
STATES_SETS = [rec.REF_TRAIT_LEVELS,
               [lev for lev, _ in orut.ELEM_TRAITS],
               [lev for lev, _, _ in orut.get_order_elem()],
               [lev for lev, _ in orut.get_order_mix_fixed_seq()]]


# # METHODS


# # MAIN
if __name__ == "__main__":
    try:
        exp_mode = min(int(argv[1]), len(STATES_SETS))
    except IndexError:
        exp_mode = 0  # Raised if no console argument was given
    except ValueError:
        exp_mode = 0  # Raised if console argument could not be cast as int

    # Connection with robot
    robot = rob.RobotInterface()
    robot.goto_neutral()

    for levels in STATES_SETS[exp_mode]:
        robot.move_traits(levels)
        input(rec.project_3d(levels))

    robot.shutdown()
