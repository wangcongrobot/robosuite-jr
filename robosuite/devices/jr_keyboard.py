"""
Driver class for Keyboard controller for JR2.
"""

import glfw
import numpy as np
from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix

class JRKeyboard(Device):
    """A Keyboard driver class to command JR2 with joint velocities."""

    def __init__(self, eef_type):
        """
        Initialize a Keyboard device.
        """
        # Set state dimensions according to eef type (dof)
        if eef_type == "hook":
          # 8 dof
          self.state = {
                        1: 0.0,
                        2: 0.0,  
                        3: 0.0,  
                        4: 0.0,  
                        5: 0.0,  
                        6: 0.0,  
                        7: 0.0,  
                        8: 0.0,  
                        }
        elif eef_type == "static":
          # 2 dof
          self.state = {
                        1: 0.0,
                        2: 0.0,  
                        }
        elif eef_type == "gripper":
          # 9 dof
          self.state = {
                        1: 0.0,
                        2: 0.0,  
                        3: 0.0,  
                        4: 0.0,  
                        5: 0.0,  
                        6: 0.0,  
                        7: 0.0,  
                        8: 0.0,  
                        9: 0.0,  
                        }
  
        self._display_controls()
        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._vel_step = 0.1 # Step size for incrementing joint velocity
    
    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("z", "Set all joint velocities to zero")
        print_command("w-s", "Increment/decrement joint velocity command")
        print_command("r", "Set velocity of current joint to 0")
        print_command("1-{}".format(len(self.state)), "Number of joint to command")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        for joint_num in self.state:
          self.state[joint_num] = 0.0

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """Returns the current state of the keyboard, a dictionary of pos, orn, grasp, and reset."""
        return self.state

    def on_press(self, window, key, scancode, action, mods):
        """
        Key handler for key presses.
        """
        if key == glfw.KEY_W:
          self.state[self.joint] += self._vel_step
        elif key == glfw.KEY_S:
          self.state[self.joint] -= self._vel_step
        elif key == glfw.KEY_R:
          self.state[self.joint]  = 0.0
  
        if key == glfw.KEY_1:
          self.joint  = 1
        elif key == glfw.KEY_2:
          self.joint  = 2
        elif key == glfw.KEY_3:
          self.joint  = 3
        elif key == glfw.KEY_4:
          self.joint  = 4
        elif key == glfw.KEY_5:
          self.joint  = 5 
        elif key == glfw.KEY_6:
          self.joint  = 6
        elif key == glfw.KEY_7:
          self.joint  = 7
        elif key == glfw.KEY_8:
          self.joint  = 8
        elif key == glfw.KEY_9:
          self.joint  = 9

    def on_release(self, window, key, scancode, action, mods):
        """
        Key handler for key releases.
        """
        # user-commanded reset
        if key == glfw.KEY_Z:
            self._reset_state = 1
            self._enabled = False
            self._reset_internal_state()
