from collections import OrderedDict
import numpy as np
import time
import pprint

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv
from robosuite.environments.jr2 import JR2Env

from robosuite.models.objects import EmptyWithGoalObject
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.robots import Baxter
from robosuite.models.tasks import DoorTask
from robosuite.models import MujocoWorldBase


class JR2Nav(JR2Env):
    """
    This class corresponds to door opening task for JR2.
    """

    def __init__(
        self,
        use_object_obs=True,
        reward_shaping=True,
        goal_offset=[0,0],
        debug_print=True,
        dist_to_door_coef=0.0,
        **kwargs
    ):
        """
        Args:
            use_object_obs (bool): if True, include object (pot) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.
  
        Inherits the JR2 environment; refer to other parameters described there.
        """

        # initialize the door
        self.goal = EmptyWithGoalObject()

        self.mujoco_objects = OrderedDict([("Goal", self.goal)])

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # goal position offset from door center
        self.goal_offset = goal_offset

        self.dist_to_door_coef = dist_to_door_coef
    
        self.debug_print = debug_print

        super().__init__(
            **kwargs
        )

    def _load_model(self):
        """
        Loads the arena and pot object.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])
        self.mujoco_objects["Goal"].set_goal_xpos(self.goal_offset[0], self.goal_offset[1])

        # load model for table top workspace
        self.model = MujocoWorldBase()
        self.mujoco_arena = EmptyArena()
        
        self.model = DoorTask(
          self.mujoco_arena,
          self.mujoco_robot,
          self.mujoco_objects,
        )
        
        self.model.place_objects([0,0,0],[1,0,0,0])
  
    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.goal_site_id = self.sim.model.site_name2id("goal")
  
        # Test prints
        #print(self.sim.model.body_names)
        #body_ids = [ self.sim.model.body_name2id(x) for x in self.sim.model.body_names]
        #print(body_ids)
        #joint_ids = [ self.sim.model.get_joint_q(x) for x in self.sim.model.joint_names]
        #print(joint_ids)
      
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.prev_dist = np.linalg.norm(self.robot_base_offset_pos[0:2] - self._goal_pos[0:2])

    def reward(self, action):
        """
        Reward function for the task.
        """
        reward = 0

        # Penalize self contacts (arm and eef with body)
        self_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms+self.mujoco_robot.gripper_contact_geoms,self.mujoco_robot.body_contact_geoms) 
        self_con_num = len(list(self_con)) > 0
        #print(self_con_num)

        # Penalize large forces
        if (abs(np.linalg.norm(self._eef_force_measurement)) > 100):
          rew_eef_force = self.force_coef
          self.large_force = True
        else:
          rew_eef_force = 0
  
        # Reward for distance to goal
        base_to_door_dist = np.linalg.norm(self.robot_base_offset_pos[0:2] - self._goal_pos[0:2])
        rew_dist_to_goal = self.dist_to_door_coef * (1 - np.tanh(base_to_door_dist))

        # Reward for progress to goal
        dist_delta = base_to_door_dist - self.prev_dist
        self.prev_dist = base_to_door_dist
        rew_progress_to_goal = np.sign(dist_delta) * rew_dist_to_goal

        reward = rew_dist_to_goal
        #reward = rew_progress_to_goal

        print("TOTAL REWARD:       {}".format(reward))
        print("distance to door:   {}".format(base_to_door_dist))
        #print("distance delta:     {}".format(dist_delta))

        return reward
    
    @property
    def _goal_pos(self):
        """ Returns position of door center in world frame """
        return self.sim.data.site_xpos[self.goal_site_id]

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
 
        # Object information
        if self.use_object_obs:
          di["goal_pos"] = self._goal_pos[0:2]

          di["object-state"] = np.concatenate(
            [
              di["goal_pos"],
            ]
          )

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = super()._check_contact()
        return collision

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        # cube is higher than the table top above a margin
        cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        table_height = self.table_full_size[2]
        return cube_height > table_height + 0.10
