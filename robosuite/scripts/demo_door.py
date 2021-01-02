import argparse
import robosuite as suite
import numpy as np
import time
import csv

from robosuite.scripts.custom_parser import custom_arg_parser, load_defaults, serialize_args

if __name__ == "__main__":
  
  custom_parser = custom_arg_parser()
  args = custom_parser.parse_args()
  load_defaults(args)
  print(args)
  print(serialize_args(args))

  env = suite.make(
    "JR2Nav",
    has_renderer=True,
    use_camera_obs=False,
    ignore_done=False,
    horizon             = args.horizon,
    control_freq=20,
    goal_offset         = args.goal_offset,
    robot_pos           = args.robot_pos,
    robot_theta         = args.robot_theta,
    debug_print         = args.print_info,
    eef_type            = args.eef_type,
    dist_to_door_coef   = args.rcoef_dist_to_door,
  )
  #env = suite.make(
  #    #"JR2StaticArmDoor",
  #    "JR2Door",
  #    has_renderer=True,
  #    use_camera_obs=False,
  #    ignore_done=False,
  #    horizon             = args.horizon,
  #    control_freq=20,
  #    door_type           = args.door_type,
  #    robot_pos           = args.robot_pos,
  #    robot_theta         = args.robot_theta,
  #    dist_to_handle_coef = args.rcoef_dist_to_handle,
  #    door_angle_coef     = args.rcoef_door_angle,
  #    handle_con_coef     = args.rcoef_handle_con,
  #    body_door_con_coef  = args.rcoef_body_door_con,
  #    self_con_coef       = args.rcoef_self_con,
  #    arm_handle_con_coef = args.rcoef_arm_handle_con,
  #    arm_door_con_coef   = args.rcoef_arm_door_con,
  #    force_coef          = args.rcoef_force,
  #    dist_to_door_coef   = args.rcoef_dist_to_door,
  #    wall_con_coef       = args.rcoef_wall_con,
  #    reset_on_large_force= args.reset_on_large_force,
  #    debug_print         = args.print_info,
  #    eef_type            = args.eef_type,
  #    door_init_qpos      = args.door_init_qpos,
  #    goal_offset         = args.goal_offset,
  #)
  
  env.reset()
  env.render()
  
  if args.eef_type == "gripper":
    action_vel = np.zeros(9)
  elif args.eef_type == "static":
    action_vel = np.zeros(2)
  else:
    action_vel = np.zeros(8)
    
  reward = 0
  while True:
    #if 0.95 <= reward <= 1.0:
    #  action_vel = np.zeros(2)
    #else:
    #  action_vel[0] = 0.75
    #  action_vel[1] = 0.75
    action_vel[0] = 0
    action_vel[1] = 0
       
    if args.eef_type == "gripper":
      action_vel[8] = 0
    obs, reward, done, _ = env.step(action_vel)
    env.render()
    
    #with open('episode-reward.csv', mode='a+') as fid:
    #  writer = csv.writer(fid, delimiter=',')
    #  writer.writerow([reward])

    #if done:
    #  quit()

    #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
    #time.sleep(0.05)
