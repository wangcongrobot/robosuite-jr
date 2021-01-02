import argparse
import sys
import numpy as np
import time
import json
import os
import csv

import robosuite as suite
from stable_baselines.common.policies import MlpPolicy,MlpLstmPolicy,MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO1
from stable_baselines import PPO2

from robosuite.wrappers import GymWrapper
from robosuite.scripts.custom_parser import custom_arg_parser, load_defaults, serialize_args

def main():

  parser = custom_arg_parser()
  args = parser.parse_args()
  load_defaults(args)
  print("Arguments:{}".format(args))
  # Create the model name with all the parameters
  
  model_dir_name = serialize_args(args)
  print("Model name: {}".format(model_dir_name))
  if args.model is not None:
    model_save_path = os.path.dirname(args.model) + "/"
    tb_save_path = model_save_path.replace("learned_models","tb_logs")
  else:
    model_save_path = "../../learned_models/" + model_dir_name + "/"
    tb_save_path = "../../tb_logs/" +  model_dir_name + "/"
  print("Model save path:{}".format(model_save_path))
  print("TB logs save path:{}".format(tb_save_path))
  final_model_path = model_save_path + "final_" + model_dir_name
  model_load_path = args.model
  show_render = args.visualize

  # Save args to json for training from checkpoints
  if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
    with open(model_save_path + "args.json", 'w+') as f:
      json.dump(vars(args),f,indent=2,sort_keys=True)

  env = GymWrapper(
      suite.make(
      "JR2Door",
      has_renderer        = show_render,
      use_camera_obs      = False,
      ignore_done         = False,
      control_freq        = args.control_freq,
      horizon             = args.horizon,
      door_type           = args.door_type,
      bot_motion          = args.bot_motion,
      robot_pos           = args.robot_pos,
      robot_theta         = args.robot_theta,
      dist_to_handle_coef = args.rcoef_dist_to_handle,
      door_angle_coef     = args.rcoef_door_angle,
      handle_con_coef     = args.rcoef_handle_con,
      body_door_con_coef  = args.rcoef_body_door_con,
      self_con_coef       = args.rcoef_self_con,
      arm_handle_con_coef = args.rcoef_arm_handle_con,
      arm_door_con_coef   = args.rcoef_arm_door_con,
      force_coef          = args.rcoef_force,
      gripper_touch_coef  = args.rcoef_gripper_touch,
      dist_to_door_coef   = args.rcoef_dist_to_door,
      wall_con_coef       = args.rcoef_wall_con,
      reset_on_large_force= args.reset_on_large_force,
      debug_print         = args.print_info,
      eef_type            = args.eef_type,
      door_init_qpos      = args.door_init_qpos,
      goal_offset         = args.goal_offset,
    )
  )
  
  if args.slurm:
    env = SubprocVecEnv([lambda: env for i in range(args.n_cpu)])
  else:
    env = DummyVecEnv([lambda: env])

  # Load the specified model, if there is one
  if args.model is not None:
    # Training from checkpoint, so need to reset timesteps for tb
    reset_num_timesteps = False
    if args.rl_alg == "ppo2":
      model = PPO2.load(model_load_path,env=env)
      print("Succesfully loaded PPO2 model")
    if args.rl_alg == "ppo1":
      model = PPO1.load(model_load_path,env=env)
      print("Succesfully loaded PPO1 model")
  else: 
    # New model, so need to reset timesteps for tb
    reset_num_timesteps = True
    if args.rl_alg == "ppo2":
      model = PPO2(
                  args.policy,
                  env,
                  verbose=args.verbose,
                  n_steps=args.n_steps,
                  nminibatches=args.minibatches,
                  noptepochs=args.opt_epochs,
                  cliprange=args.clip_range,
                  ent_coef=args.ent_coef,
                  tensorboard_log=tb_save_path,
                  #full_tensorboard_log=True
                  )

    elif args.rl_alg == "ppo1":
      model = PPO1(
                  args.policy,
                  env,
                  verbose=args.verbose,
                  timesteps_per_actorbatch=args.n_steps,
                  optim_epochs=args.opt_epochs,
                  tensorboard_log=tb_save_path,
                  )
  if args.replay:
    # Replay a policy
    obs = env.reset()
    count = 0
    with open('episode-reward.csv', mode='w') as fid:
      writer = csv.writer(fid, delimiter=',')
      writer.writerow("reward")
    while(count < 1000):
      env.render()
      count += 1
      print(count)
    while True:
      if args.model is None:
        print("Error: No model has been specified")
      action, _states = model.predict(obs,deterministic=True)
      #print("action {}".format(action))
      obs, reward, done, info = env.step(action)
      env.render()
      #print(obs)
      #print(env.sim.data.qpos[env._ref_joint_vel_indexes])
      #time.sleep(0.1)

      with open('episode-reward.csv', mode='a') as fid:
        writer = csv.writer(fid, delimiter=',')
        writer.writerow(reward)

      #if done:
      #  quit()
  else:
    # Train
    model.learn(
                total_timesteps = args.total_timesteps,
                save_dir = model_save_path,
                render=show_render,
                reset_num_timesteps=reset_num_timesteps,
                )

    model.save(final_model_path)
  
    print("Done training")
    obs = env.reset()

if __name__ == "__main__":
  main()

