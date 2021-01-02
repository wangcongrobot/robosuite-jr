from collections import OrderedDict
import numpy as np
import time
import pprint

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv
from robosuite.environments.jr2 import JR2Env

from robosuite.models.objects import DoorPullNoLatchObject, DoorPullWithLatchObject, DoorPullNoLatchRoomObject, DoorPullNoLatchRoomWideObject
from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.robots import Baxter
from robosuite.models.tasks import DoorTask
from robosuite.models import MujocoWorldBase


class JR2Door(JR2Env):
    """
    This class corresponds to door opening task for JR2.
    """

    def __init__(
        self,
        use_object_obs=True,
        reward_shaping=True,
        door_type="dpnl",
        door_pos = [1.3,-0.05,1.0],
        door_quat = [1, 0, 0, -1],
        dist_to_handle_coef=1.0,
        door_angle_coef=1.0,
        handle_con_coef=1.0,
        body_door_con_coef=0.0,
        self_con_coef=0.0,
        arm_handle_con_coef=0.0,
        arm_door_con_coef=0.0,
        gripper_touch_coef=0.0,
        dist_to_door_coef=0.0,
        wall_con_coef=0.0,
        force_coef=0.0,
        debug_print=False,
        door_init_qpos=0.0,
        goal_offset=[0,0],
        **kwargs
    ):
        """
        Args:
            use_object_obs (bool): if True, include object (pot) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.
  
            door_type (str): type of door (pull no latch, pull with latch, push no latch, push with latch)
    
            door_pos ([x,y,z]): position of door
   
            door_quat ([w,x,y,z]): quaternion of door

            dist_to_handle_coef: reward coefficient for eef distance to handle

            door_angle_coef: reward coefficient for angle of door
  
            handle_con_coef: reward coefficient for eef contact with handle

            body_door_con_coef: reward coefficent to penalize body contact with door

            self_con_coef: reward coefficient to penalize self collisions

            arm_handle_con_coef: reward coefficient to penalize collisions with arm and handle
      
            debug_print: True if printing debug information

        Inherits the JR2 environment; refer to other parameters described there.
        """

        # initialize the door
        if (door_type == "dpnl"):
         self.door = DoorPullNoLatchObject()
        elif (door_type == "dpwl"):
         self.door = DoorPullWithLatchObject()
        elif (door_type == "dpnlr"):
         self.door = DoorPullNoLatchRoomObject()
        elif (door_type == "dpnlrw"):
         self.door = DoorPullNoLatchRoomWideObject()

        self.door_type = door_type
        self.mujoco_objects = OrderedDict([("Door", self.door)])

        self.door_pos = door_pos
        self.door_quat = door_quat

        # Door hinge initial pos
        self.door_init_qpos = door_init_qpos
  
        # goal position offset from door center
        self.goal_offset = goal_offset

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.dist_to_handle_coef = dist_to_handle_coef
        self.door_angle_coef = door_angle_coef
        self.handle_con_coef = handle_con_coef
        self.body_door_con_coef = body_door_con_coef
        self.self_con_coef = self_con_coef
        self.arm_handle_con_coef = arm_handle_con_coef
        self.arm_door_con_coef  = arm_door_con_coef
        self.force_coef = force_coef
        self.gripper_touch_coef = gripper_touch_coef
        self.dist_to_door_coef = dist_to_door_coef
        self.wall_con_coef = wall_con_coef

        # Reset prev door hinge angle
        self.door_angle_prev = self.door_init_qpos

        self.debug_print = debug_print
        print('debug print {}'.format(self.debug_print))

        super().__init__(
            **kwargs
        )

    def _load_model(self):
        """
        Loads the arena and pot object.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])
        self.mujoco_objects["Door"].set_goal_xpos(self.goal_offset[0], self.goal_offset[1])

        # load model for table top workspace
        self.model = MujocoWorldBase()
        self.mujoco_arena = EmptyArena()
        
        self.model = DoorTask(
          self.mujoco_arena,
          self.mujoco_robot,
          self.mujoco_objects,
        )
        
        self.model.place_objects(self.door_pos,self.door_quat)
  
    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.door_body_id = self.sim.model.body_name2id("door")
        self.door_latch_id = self.sim.model.body_name2id("latch")
        self.door_handle_site_id = self.sim.model.site_name2id("door_handle")
        self.door_hinge_joint_id = self.sim.model.get_joint_qpos_addr("door_hinge")
        self.door_center_site_id = self.sim.model.site_name2id("door_center")
        self.goal_site_id = self.sim.model.site_name2id("goal")
  
        # Test prints
        #print(self.sim.model.body_names)
        #body_ids = [ self.sim.model.body_name2id(x) for x in self.sim.model.body_names]
        #print(body_ids)
        #joint_ids = [ self.sim.model.get_joint_q(x) for x in self.sim.model.joint_names]
        #print(joint_ids)
      
        # Wall references, if there are walls
        if (self.door_type == "dpnlr"):
          self.wall_geom_id = []
          for wall_geom in self.mujoco_objects["Door"].wall_contact_geoms:
            self.wall_geom_id.append(self.sim.model.geom_name2id(wall_geom))
          #print(self.wall_geom_id)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        
        # Reset door hinge angle
        self.sim.data.qpos[self.door_hinge_joint_id] = np.asarray([self.door_init_qpos])

        # Reset prev door hinge angle
        self.door_angle_prev = self.door_init_qpos

    def reward(self, action):
        """
        Reward function for the task.
        """
        reward = 0

        # Distance to door
        distance_to_handle = self._eef_distance_to_handle
        #print("distance to door: {}".format(distance_to_handle))
        #print("R: distance to door: {}".format(distance_to_handle))

        # Angle of door body (in door object frame)
        door_hinge_angle = self._door_hinge_pos

        # Penalize self contacts (arm and eef with body)
        self_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms+self.mujoco_robot.gripper_contact_geoms,self.mujoco_robot.body_contact_geoms) 
        self_con_num = len(list(self_con)) > 0
        #print(self_con_num)

        # eef contact with door handle
        door_handle_con = self.find_contacts(self.mujoco_robot.gripper_contact_geoms,self.mujoco_objects["Door"].handle_contact_geoms)
        door_handle_con_num = len(list(door_handle_con)) > 0
        #print("eef handle con num {}".format(door_handle_con_num))
      
        # Body to door contacts
        body_door_con = self.find_contacts(self.mujoco_robot.body_contact_geoms, self.mujoco_objects["Door"].door_contact_geoms + self.mujoco_objects["Door"].handle_contact_geoms)
        body_door_con_num = len(list(body_door_con)) > 0
        #print("body door con num {}".format(body_door_con_num))

        # If we want to penalize body door contact,
        # Check for consecutive body to door contacts
        if self.body_door_con_coef < 0:
          if body_door_con_num > 0:
            self.consecutive_body_door_con += 1
          else:
            self.consecutive_body_door_con = 0

        # Arm to door contacts
        arm_door_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms + self.mujoco_robot.gripper_contact_geoms, self.mujoco_objects["Door"].door_contact_geoms)
        arm_door_con_num = len(list(arm_door_con)) > 0
        #print("arm door con num {}".format(arm_door_con_num))

        # Arm links to door handle contacts 
        arm_handle_con = self.find_contacts(self.mujoco_robot.arm_contact_geoms, self.mujoco_objects["Door"].handle_contact_geoms)
        arm_handle_con_num = len(list(arm_handle_con)) > 0
        #print(arm_handle_con_num)

        # Penalize large forces
        if (abs(np.linalg.norm(self._eef_force_measurement)) > 100):
          rew_eef_force = self.force_coef
          if self.debug_print:
            print("LARGE FORCE {}".format(self._eef_force_measurement))
            time.sleep(1)
          self.large_force = True
        else:
          rew_eef_force = 0
  
        # If JR has gripper, check touch sensor in gripper
        if self.eef_type == "gripper":
          if self._gripper_touch_measurement>0:
            rew_gripper_touch = self.gripper_touch_coef
          else:
            rew_gripper_touch = 0
        else:
            rew_gripper_touch = 0

        #print("handle xpos: {}".format(self._door_handle_xpos))

        # Reward for going through door
        base_to_door_dist = np.linalg.norm(self.robot_base_offset_pos[0:2] - self._goal_pos[0:2])
        rew_dist_to_door = self.dist_to_door_coef * (1 - np.tanh(base_to_door_dist))

        # Check contact with walls
        if (self.door_type == "dpnlr" or self.door_type == "dpnlrw"):
          wall_con = self.find_contacts(self.mujoco_robot.gripper_contact_geoms + self.mujoco_robot.arm_contact_geoms + self.mujoco_robot.body_contact_geoms, self.mujoco_objects["Door"].wall_contact_geoms)
          wall_con_num = len(list(wall_con)) > 0
          rew_wall_con = self.wall_con_coef * wall_con_num

          # Check for consecutive wall contact
          if wall_con_num > 0:
            self.consecutive_wall_con += 1
          else:
            self.consecutive_wall_con = 0
        else:
          rew_wall_con = 0.0

        # New reward for door angle:
        # negative if door angle is decreased, positive if it is increased
        door_angle_delta = self._door_hinge_pos - self.door_angle_prev        
        self.door_angle_prev = self._door_hinge_pos
        door_angle_delta_sign = np.sign(door_angle_delta)

        rew_dist_to_handle = self.dist_to_handle_coef * (1 - np.tanh(distance_to_handle))
        rew_door_angle     = self.door_angle_coef * door_angle_delta_sign
        #rew_door_angle     = self.door_angle_coef * door_hinge_angle
        rew_handle_con     = self.handle_con_coef * door_handle_con_num
        rew_body_door_con  = self.body_door_con_coef * body_door_con_num
        rew_self_con       = self.self_con_coef * self_con_num
        rew_arm_handle_con = self.arm_handle_con_coef * arm_handle_con_num
        rew_arm_door_con   = self.arm_door_con_coef * arm_door_con_num
        #print(self.arm_handle_con_coef)

        reward = (rew_dist_to_handle + 
                  rew_door_angle + 
                  rew_handle_con + 
                  rew_body_door_con + 
                  rew_self_con + 
                  rew_eef_force + 
                  rew_arm_door_con +  
                  rew_gripper_touch + 
                  rew_dist_to_door + 
                  rew_wall_con + 
                  rew_arm_handle_con)

        #print("TOTAL REWARD:       {}".format(reward))
        #print("distance to door:   {}".format(base_to_door_dist))
        if self.debug_print:
          print("TOTAL REWARD:       {}".format(reward))
          print("rew_dist_to_handle: {}".format(rew_dist_to_handle))
          print("rew_door_angle:     {}".format(rew_door_angle))
          print("rew_handle_con:     {}".format(rew_handle_con))
          print("body_door_con:  {}".format(body_door_con_num))
          print("arm_door_con:   {}".format(arm_door_con_num))
          print("rew_body_door_con:  {}".format(rew_body_door_con))
          print("rew_arm_door_con:   {}".format(rew_arm_door_con))
          print("rew_arm_handle_con: {}".format(rew_arm_handle_con))
          print("rew_self_con:       {}".format(rew_self_con))
          print("rew_wall_con:       {}".format(rew_wall_con))
          print("rew_eef_force:      {}".format(rew_eef_force))
          print("rew_gripper_touch:  {}".format(rew_gripper_touch))
          print("rew_dist_to_door:   {}".format(rew_dist_to_door))
          print("distance to door:   {}".format(base_to_door_dist))

        return reward
    
    @property
    def _door_xpos(self):
        """ Returns the position of the door """
        return self.sim.data.body_xpos[self.door_body_id]
    
    @property
    def _door_handle_xpos(self):
        """ Returns position of door handle target site """
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _door_center_pos(self):
        """ Returns position of door center in world frame """
        return self.sim.data.site_xpos[self.door_center_site_id]

    @property
    def _goal_pos(self):
        """ Returns position of door center in world frame """
        return self.sim.data.site_xpos[self.goal_site_id]

    @property
    def _door_latch_xquat(self):
        """ Returns angle of door latch """
        return self.sim.data.body_xquat[self.door_latch_id]

    @property
    def _door_hinge_pos(self):
        """ Returns angle of door hinge joint """
        return self.sim.data.qpos[self.door_hinge_joint_id]

    @property
    def _eef_distance_to_handle(self):
        """ Returns vector from robot to door handle """
        dist = np.linalg.norm(self._door_handle_xpos - self._r_eef_xpos )
        return dist 

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    @property
    def _r_gripper_to_handle(self):
        """Returns vector from the right gripper to the handle."""
        return self._handle_2_xpos - self._r_eef_xpos

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
          # position and rotation of object in world frame
          door_pos = self.sim.data.body_xpos[self.door_body_id]
          door_quat = T.convert_quat(self.sim.data.body_xquat[self.door_body_id], to="xyzw")
          #print("door pos: {}".format(door_pos))

          #di["door_pos"] = door_pos[0:2]
          #di["door_quat"] = door_quat
          di["hinge_theta"] = np.array([self._door_hinge_pos])
          di["door_handle_pos"] = self._door_handle_xpos 
          #di["handle_quat"] =  self._door_latch_xquat
          di["goal_pos"] = self._goal_pos[0:2]
          #print(di["handle_quat"])

          # If JR has gripper, check touch sensor in gripper and add to observation
          if self.eef_type == "gripper":
            if self._gripper_touch_measurement>0:
              di["gripper_touch"] = np.array([1])
              #print("object state obs {}".format(di["gripper_touch"]))
            else:
              di["gripper_touch"] = np.array([0])

            di["object-state"] = np.concatenate(
              [
                #di["door_pos"],
                di["hinge_theta"],
                di["door_handle_pos"],
                #di["handle_quat"],
                di["gripper_touch"],
                di["goal_pos"],
              ]
            )
          else:
            di["object-state"] = np.concatenate(
              [
                #di["door_pos"],
                di["hinge_theta"],
                di["door_handle_pos"],
                #di["handle_quat"],
                di["goal_pos"],
              ]
            )

        if self.debug_print:
          pp = pprint.PrettyPrinter(indent=1)
          print("Observation")
          pp.pprint(di)
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
