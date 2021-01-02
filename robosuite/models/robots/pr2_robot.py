import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class PR2(Robot):
    """PR2."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/PR2/pr2.xml"))

        self.bottom_offset = np.array([0, 0, -0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base_footprint']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        #print(self.worldbody.find("./body/joint").get("name"))
        #return [self.worldbody.find("./body/joint").get("name")]
        return [
                "rootx",
                "rooty",
                "rootwz",
                "torso_lift_joint",
                "r_shoulder_pan_joint",
                "r_shoulder_lift_joint",
                "r_upper_arm_roll_joint",
                "r_elbow_flex_joint",
                "r_forearm_roll_joint",
                "r_wrist_flex_joint",
                "r_wrist_roll_joint",
                "r_gripper_l_finger_joint",
                "r_gripper_l_finger_tip_joint",
                "r_gripper_r_finger_joint",
                "r_gripper_r_finger_tip_joint",
                "l_shoulder_pan_joint",
                "l_shoulder_lift_joint",
                "l_upper_arm_roll_joint",
                "l_elbow_flex_joint",
                "l_forearm_roll_joint",
                "l_wrist_flex_joint",
                "l_wrist_roll_joint",
                "l_gripper_l_finger_joint",
                "l_gripper_l_finger_tip_joint",
                "l_gripper_r_finger_joint",
                "l_gripper_r_finger_tip_joint",
               ]

    @property
    def init_qpos(self):
        #return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
        return np.zeros(26)
