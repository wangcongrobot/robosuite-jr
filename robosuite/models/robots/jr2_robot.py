import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class JR2(Robot):
    """JR2."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/jr2/jr2_with_arm.xml"))

        self.bottom_offset = np.array([0, 0, 0])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='base_footprint']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    @property
    def dof(self):
        return 8

    @property
    def arm_joints(self):
        return [
                "m1n6s200_joint_1",
                "m1n6s200_joint_2",
                "m1n6s200_joint_3",
                "m1n6s200_joint_4",
                "m1n6s200_joint_5",
                "m1n6s200_joint_6",
               ]

    @property
    def base_joints(self):
        return [
                "left_wheel",
                "right_wheel",
               ]

    @property
    def init_arm_qpos(self):
        # extended arm
        #pos = np.array([-1.71255116,2.80531801,3.29199776,1.71876122,0.04361478,-0.00627881])

        # bent arm
        pos = np.array([-1.66453506,1.98583623,2.01604606,1.71591313,0.02474466,-0.00216071])
      
        # arm turned in
        #pos = np.array([-0.9864941,1.68345062,1.80218911,-5.26739192,-0.01367158,-0.1648811])
  
        # angled body
        #pos = np.array([-0.89198715,1.7414945,1.92991576,-5.45929098,0.38113745,0.07569197])

        # 45 degree body - arm
        #pos = np.array([-1.99673054,1.22329899,1.67109673,5.49132849,-0.66292144,-2.96626974])
        return pos
    
    @property
    def visualization_sites(self):
        return ["r_grip_site",]

    @property
    def body_contact_geoms(self):
        return[
          "body",
          "neck",
          "head",
          "front_caster",
          "rear_caster",
          "l_wheel_link",
          "r_wheel_link",
        ]
  
    @property
    def arm_contact_geoms(self):
        return[
          "armlink_base",
          "armlink_2",  
          "armlink_3",  
          "armlink_5",  
          "armlink_6",  
        ]

    @property
    def gripper_contact_geoms(self):
        return[
          "fingertip_2",
          "fingertip_2_hook",
        ]
