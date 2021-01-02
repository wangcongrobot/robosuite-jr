from collections import OrderedDict
import numpy as np
import time

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import JR2, JR2Gripper, JR2StaticArm


class JR2Env(MujocoEnv):
    """Initializes a JR robot environment."""

    def __init__(
        self,
        use_indicator_object=False,
        rescale_actions=True,
        bot_motion="mmp",
        reset_on_large_force=False,
        robot_pos=[0,0,0],
        robot_theta=0,
        eef_type="gripper",
        **kwargs
    ):
        """
        Args:
            bot_motion (str): "static" for static base, "mmp" for mobile base

            use_indicator_object (bool): if True, sets up an indicator object that 
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in 
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes 
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes 
                in camera. False otherwise.

            control_freq (float): how many control signals to receive 
                in every second. This sets the amount of simulation time 
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a 
                rendered image.

            camera_name (str): name of camera to be rendered. Must be 
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.

            reset_on_large_force (bool): False if environment should not reset when eef applies large force
      
            init_distance (str): Intial qpos specifier for robot
        """
        self.use_indicator_object = use_indicator_object
        self.rescale_actions = rescale_actions
        self.bot_motion = bot_motion
        self.large_force = False
        self.reset_on_large_force = reset_on_large_force
        self.init_robot_pos = robot_pos
        self.init_robot_theta = robot_theta
        self.eef_type = eef_type

        self.consecutive_body_door_con = 0
        self.consecutive_wall_con = 0
        self.contact_thresh = 10

        super().__init__(**kwargs)

    def _load_model(self):
        """Loads robot and optionally add grippers."""
        super()._load_model()
        if self.eef_type == "hook":
          self.mujoco_robot = JR2()
        elif self.eef_type == "gripper":
          self.mujoco_robot = JR2Gripper()
        elif self.eef_type == "static":
          self.mujoco_robot = JR2StaticArm()
        else:
          print("Error: Invalid EEF type")

    def _reset_internal(self):
        """Resets the pose of the arm and grippers."""
        if self.debug_print:
          print("\nRESETTING ENVIRONMENT")
        super()._reset_internal()
      
        # Calculate robot initial quaternion from theta
        qz = np.sin(self.init_robot_theta/2)
        qw = np.cos(self.init_robot_theta/2)
        # Reset position and angle of base 
        self.sim.data.qpos[self._ref_free_joint_pos_indexes[0:3]] = self.init_robot_pos
        self.sim.data.qpos[self._ref_free_joint_pos_indexes[3:7]] = [qw, 0, 0, qz]

        #if self.eef_type == "static":
        # Reset arm 
        #self.sim.data.qpos[0:7] = self.mujoco_robot.init_base_qpos
        self.sim.data.qpos[self._ref_arm_joint_pos_indexes] = self.mujoco_robot.init_arm_qpos

      
        self.large_force = False
        self.consecutive_body_door_con = 0
        self.consecutive_wall_con = 0
        #else:
        #  self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos(self.init_distance)
        #  self.large_force = False

    def _get_reference(self):
        """Sets up references for robots, grippers, and objects."""
        super()._get_reference()

        self._ref_free_joint_pos_indexes = np.linspace(self.sim.model.get_joint_qpos_addr("root")[0],self.sim.model.get_joint_qpos_addr("root")[1]-1,self.sim.model.get_joint_qpos_addr("root")[1]-self.sim.model.get_joint_qpos_addr("root")[0]).astype(int)
        
        self._ref_free_joint_vel_indexes = np.linspace(self.sim.model.get_joint_qvel_addr("root")[0],self.sim.model.get_joint_qvel_addr("root")[1]-1,self.sim.model.get_joint_qvel_addr("root")[1]-self.sim.model.get_joint_qvel_addr("root")[0]).astype(int)
      
        # indices for base and arm joint qpos and qvel
        self.robot_base_joints = list(self.mujoco_robot.base_joints)
        self.robot_arm_joints = list(self.mujoco_robot.arm_joints)

        self._ref_base_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_base_joints
        ]
        self._ref_base_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_base_joints
        ]
        self._ref_arm_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_arm_joints
        ]
        self._ref_arm_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_arm_joints
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]
        self.r_grip_site_id = self.sim.model.site_name2id("r_grip_site")
        self.base_offset_site_id = self.sim.model.site_name2id("base_offset")

    # Note: Overrides super
    def _pre_action(self, action):
      # If robot has hook as eef, action is an 8-dim vector (x,theta,arm joint velocities)
      # If robot has gripper, action is a 9-dim vector
      # Copy the action to a list
      new_action = action.copy().tolist()
      #print("Policy action {}".format(new_action))
      if self.debug_print:
        print("\nTimestep: {}".format(self.timestep))
        print("Robot pre_action info")
        print("Policy action {}".format(new_action))

      # If JR has a binary gripper, process the fingers' actions
      # Add an additional value to the end of action, for the second finger
      if self.eef_type == "gripper":
        new_action.append(action[8])
        # gripper state: 1 is open, -1 is closed
        #gripper_state = action[8]
        #if action[8] == 0:
        #  is_open = True
        #else:
        #  is_open = False
        #action[8] = 0.0
        #new_action.append(0.0)

        # set position of gripper
        #if is_open:
        #  self.sim.data.qpos[10] = 0.0
        #  self.sim.data.qpos[9] = 0.0
        #else:
        #  self.sim.data.qpos[10] = 0.638
        #  self.sim.data.qpos[9] = 0.638

      # Scale and clip the policy actions, while preserving Gaussian shape
      if self.rescale_actions:
          # Scale normalized action to control ranges
          ctrl_range = self.sim.model.actuator_ctrlrange
          bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
          weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
          applied_action = bias + weight * new_action
      
          # Clip
          new_action = np.clip(new_action, -1, 1)
      else:
          applied_action = new_action

      # If arm is only static, keep the arm joints at zero velocity
      if self.eef_type == "static":
        #self.sim.data.qpos[self._ref_arm_joint_pos_indexes] = self.mujoco_robot.init_arm_qpos
        self.sim.data.qvel[self._ref_arm_joint_vel_indexes] = 0.0
        self.sim.data.ctrl[:] = applied_action
      else:
        if (self.bot_motion == "static"):
          self.sim.data.qvel[0] = 0.0
          self.sim.data.qvel[1] = 0.0
          self.sim.data.qvel[2] = 0.0

      applied_action[0] = applied_action[0] * 1
      applied_action[1] = applied_action[1] * 1

      # Apply the action
      self.sim.data.ctrl[:] = applied_action

      if self.debug_print:
        print("applied action {}".format(applied_action))
        print("actual base qvels {}".format(self.sim.data.qvel[self._ref_base_joint_vel_indexes]))
        print("actual arm qvels {}".format(self.sim.data.qvel[self._ref_arm_joint_vel_indexes]))
        print("entire qvel {}".format(self.sim.data.qvel)) 

      # gravity compensation
      self.sim.data.qfrc_applied[
          self._ref_arm_joint_vel_indexes
      ] = self.sim.data.qfrc_bias[self._ref_arm_joint_vel_indexes]
      self.sim.data.qfrc_applied[
          self._ref_base_joint_vel_indexes
      ] = self.sim.data.qfrc_bias[self._ref_base_joint_vel_indexes]

    def _post_action(self, action):
        """Optionally performs gripper visualization after the actions."""
        #ret = super()._post_action(action)
        reward = self.reward(action)

        # reset after surpassing contact threshold
        reset_con = self.consecutive_body_door_con > self.contact_thresh or \
                    self.consecutive_wall_con > self.contact_thresh

        #if self.consecutive_body_door_con > 0 or \
        #            self.consecutive_wall_con > 0:
        #  print("Consecutive contacts {} {}".format(self.consecutive_body_door_con \
        #                                         ,self.consecutive_wall_con))

        if self.reset_on_large_force:
          self.done = ((self.large_force) or (self.timestep >= self.horizon) or reset_con) and not self.ignore_done 
        else:
          self.done = ((self.timestep >= self.horizon) or reset_con) and not self.ignore_done 
          
        return reward, self.done, {}

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-centric information.
        """
        di = super()._get_observation()
        self._check_contact()

        di["base_pos"] = self.robot_base_pos.flatten()
        di["base_theta"] = self.robot_base_theta.flatten()

        di["base_vel"] = np.array([self.sim.data.qvel[x] for x in self._ref_free_joint_vel_indexes[0:2]] + [self.sim.data.qvel[self._ref_free_joint_vel_indexes[5]]])
        #di["base_vel_old"] = np.array([self.sim.data.qvel[x] for x in range(7)])

        # Arm joint info
        di["arm_joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_arm_joint_pos_indexes]
        )
        di["arm_joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_arm_joint_vel_indexes]
        )

        # Wheel velocities
        di["wheel_joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_base_joint_vel_indexes]
        )

        di["r_eef_xpos"] = self._r_eef_xpos
        di["r_eef_xquat"] = self._r_eef_xquat
        #di["robot_base_pos"] = self.robot_base_pos
        #di["robot_base_theta"] = self.robot_base_theta
  
        robot_states = [
            di["base_pos"],
            di["base_theta"],
            di["base_vel"],
            #di["arm_joint_pos"],
            #di["arm_joint_vel"],
            #di["wheel_joint_vel"],
            #di["r_eef_xpos"],
            #di["r_eef_xquat"],
        ]
    
        di["robot-state"] = np.concatenate(robot_states)
        
        if self.debug_print:
          print("EEF force/torque {}/{}".format(np.linalg.norm(self._eef_force_measurement),self._eef_torque_measurement))
          #print("Robot pose {}".format(self.robot_pose_in_world))
          #print("Robot state obs {}".format(di))

        return di

    @property
    def dof(self):
        """Returns the DoF of the robot (with grippers)."""
        dof = self.mujoco_robot.dof
        return dof

    @property
    def theta_w(self):
        """Returns theta of robot in world frame"""
        theta = self.sim.data.qpos[self._rootwz_ind]
        return theta

    @property
    def robot_pose_in_world(self):
        pos_in_world = self.sim.data.get_body_xpos("base_footprint")
        rot_in_world = self.sim.data.get_body_xmat("base_footprint").reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)
        return pose_in_world

    @property
    def robot_base_offset_pos(self):
        """
        Base position of robot in world frame
        """
        return self.sim.data.site_xpos[self.base_offset_site_id]

    @property
    def robot_base_pos(self):
        """
        Base position of robot in world frame
        """
        return self.robot_pose_in_world.flatten()[[3,7]]

    @property
    def robot_base_theta(self):
        """
        Base angle of robot in world frame (around z axis)
        """
        abs_theta = np.arccos(self.robot_pose_in_world.flatten()[0])
        # determine quadrant of angle
        sin_sign = np.sign(self.robot_pose_in_world[1][0])
        
        if sin_sign != 0:
          return abs_theta * sin_sign
        else:
          return abs_theta
            

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose of that
        object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def action_spec(self):
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
      
        return low, high

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame as a numpy
        array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes[:7]]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes[:7]]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot. 
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_quat(self):
        """
        Returns eef orientation of right hand in base from of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """Returns a numpy array of joint positions (angles), of dimension 14."""
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """Returns a numpy array of joint (angular) velocities, of dimension 14."""
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    @property
    def _r_eef_xpos(self):
        """Returns the position of the right hand site in world frame."""
        return self.sim.data.site_xpos[self.r_grip_site_id]

    @property
    def _r_eef_xquat(self):
       """Returns the position of the right hand site in world frame."""
       return T.mat2quat(self.sim.data.site_xmat[self.r_grip_site_id].reshape((3,3)))

    @property
    def _eef_force_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("force_ee")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _eef_torque_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("torque_ee")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _gripper_touch_measurement(self):
        """Returns measurement of touch sensor in robot gripper."""
        sensor_id = self.sim.model.sensor_name2id("touch_gripper")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _l2_force_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("force_2")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    @property
    def _l3_force_measurement(self):
        """Returns sensor measurement."""
        sensor_id = self.sim.model.sensor_name2id("force_3")
        return self.sim.data.sensordata[sensor_id*3 : sensor_id*3 + 3]

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Using defaults.
        """
        pass

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False
