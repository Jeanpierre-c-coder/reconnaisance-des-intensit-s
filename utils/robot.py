# -*- coding: utf-8 -*-
"""
Toolbox and class for sending motor commands to the robot
By Sébastien Mick
"""
# # IMPORTS
# - Built-in
from time import sleep
# - Third-party
import numpy as np
# - Local
from .maestro import Controller


# # CONSTANTS
# Servomotor index table
# 1:  Forehead
# 2:  Inner right eyebrow
# 3:  Outer right eyebrow
# 4:  Eyes tilt
# 5:  Left eye pan
# 6:  Right eye pan
# 7:  Right lip corner height
# 9:  Mouth bottom height
# 11: Right lip corner stretch
# 12: Inner left eyebrow
# 13: Outer left eyebrow
# 14: Left mouth corner height
# 15: Left mouth corner stretch

# Motor commands for each possible state of each group of traits
ELEM_SERVO_POS = [[{1: 1650, 2: 2000, 3: 1450, 12: 900, 13: 1500},    # EB:angry
                   {1: 1550, 2: 1500, 3: 1450, 12: 1400, 13: 1500},   # EB:neutral
                   {1: 1150, 2: 1900, 3: 1150, 12: 1000, 13: 1800}],  # EB:surprised
                  [{7: 2150, 14: 1400},   # LC:sad
                   {7: 1600, 14: 2025},   # LC:neutral
                   {7: 1200, 14: 2500}],  # LC:happy
                  [{9: 1950, 11: 1200, 15: 1225},   # MO:angry
                   {9: 1800, 11: 1400, 15: 1350},   # MO:neutral
                   {9: 1500, 11: 1500, 15: 1450}]]  # MO:surprised
# Motor commands for actuators that remain static
NEUTRAL_SERVO_POS = {4: 870, 5: 1600, 6: 1575}
for grp_pos in ELEM_SERVO_POS:
    NEUTRAL_SERVO_POS.update(grp_pos[1])


# # METHODS


# # CLASS
class RobotInterface:
    """
    Basic interface with robot, providing high-level features to interact with
    motors through commands to Pololu Mini Maestro board
    """

    def __init__(self, port=None):
        """
        Open serial connection with Maestro board at given serial port
        """
        dev = "/dev/ttyACM0" if port is None else port
        self.servos = Controller(dev)
        self.set_speed(40)

    def set_targets(self, targets):
        """
        Send multiple commands to motors
        """
        for motor_id, target in targets.items():
            self.servos.set_target(motor_id, target * 4)
            # A relation by a factor 4 was empirically noted between equivalent
            # commands sents from Maestro Controller and maestro.py
            # Possible origin would be an extra two-bit shift in the byte
            # sequence written by maestro.py on serial port

    def move_traits(self, levels):
        """
        Move eyebrows and mouth motors to produce given facial expression, as
        described by activation levels of trait groups
        """
        commands = []
        for elem_pos, group_lev in zip(ELEM_SERVO_POS, levels):
            act = np.argmax(group_lev)
            targets = elem_pos[act]
            if act == 1:  # Queue commands for neutral activation first
                commands.insert(0, targets)
            else:
                commands.append(targets)
            # Quite suboptimal quick fix: better solutions exist, typically
            # to avoid sending two commands to the same motor
        for targets in commands:
            self.set_targets(targets)

    def goto_neutral(self):
        """
        Bring all registered motors to their neutral position
        """
        self.set_targets(NEUTRAL_SERVO_POS)

    def set_speed(self, speed):
        """
        Set max moving speed of all motors
        """
        for motor_id in NEUTRAL_SERVO_POS.keys():
            self.servos.set_speed(motor_id, speed)

    def turn_off(self):
        """
        Turn off all registered motors
        """
        for motor_id in NEUTRAL_SERVO_POS.keys():
            self.servos.set_target(motor_id, 0)
            # 0 is not a target position but a command to turn off servomotors

    def shutdown(self):
        """
        Bring all registered motors to their neutral positions, turn them off
        and close serial connection to board
        """
        self.goto_neutral()
        sleep(0.5)
        self.turn_off()
        self.servos.close()


# # MAIN
if __name__ == "__main__":
    robot = RobotInterface()
    robot.shutdown()
