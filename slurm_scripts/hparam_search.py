#!/usr/bin/env python
import os
import time
import numpy as np
import csv

def main():
  print("Simple navigation test")
  # Modify the slurm script with corresponding initial conditions
  
  CONFIG_FILE  = "~/robosuite/robosuite/scripts/default_door.json"
  
  BOT_MOTION           = "mmp"
  DOOR_TYPE            = "na"
  EEF_TYPE             = "static"
  DOOR_THETA           = 2.0
  
  NSTEPS               = 8192
  HORIZON              = 4096
  OPT_EPOCHS           = 10
  MINIBATCHES          = 4
  ENT_COEF             = 0.0
  VF_COEF              = 0.5
  CLIP_RANGE           = 0.2
  GAMMA                = 0.99
  LAM                  = 0.95
  
  RCOEF_SELF_CON       = 0
  RCOEF_FORCE          = -10
  RCOEF_DIST_TO_DOOR   = 1
  
  RESET_ON_LARGE_FORCE = "t"

  ROBOT_THETA          = 0.0
  ROBOT_POS_X          = 0.0
  ROBOT_POS_Y          = 0.0
  GOAL_OFFSET_X        = 3.5
  GOAL_OFFSET_Y        = 0.0

  counter = 0
  
  # Experiments CSV header
  with open("hparam_experiments.csv", mode="a") as csv_id:
    writer = csv.writer(csv_id, delimiter=",", lineterminator="\n")
    writer.writerow(["job_name", "robot_pos_x", "robot_pos_y", "init_robot_theta", "goal_pos_x", "goal_pos_y", "clip_range", "opt_epochs", "minibatches", "gamma", "lam", "ent_coef", "vf_coef"])

  for GOAL_OFFSET_Y in [0.0, 1.0]:
    for CLIP_RANGE in [0.1,0.2]:
      for OPT_EPOCHS in [10,20,30]:
        for MINIBATCHES in [4, 15]:
          for GAMMA in [0.99]:
            for LAM in [0.95]:
              for ENT_COEF in [0.0, 0.01]:
                for VF_COEF in [0.5]:
                  # Create job name
                  job_name = "{}".format(counter) + "_hparam"
                  DESCRIPTION          = job_name

                  # Format goal offset string
                  GOAL_OFFSET = "{} {}".format(GOAL_OFFSET_X,GOAL_OFFSET_Y)
                  # Format robot pos string
                  ROBOT_POS = "{} {} 0.0".format(ROBOT_POS_X,ROBOT_POS_Y)
  
                  # Write params to csv
                  with open("hparam_experiments.csv", mode="a") as csv_id:
                    writer = csv.writer(csv_id, delimiter=",", lineterminator="\n")
                    writer.writerow([DESCRIPTION, ROBOT_POS_X, ROBOT_POS_Y, ROBOT_THETA, GOAL_OFFSET_X, GOAL_OFFSET_Y, CLIP_RANGE, OPT_EPOCHS, MINIBATCHES, GAMMA, LAM, ENT_COEF, VF_COEF])
  
                  # Format sbatch command
                  with open("nav_test.sbatch", "r") as sbatch_template:
                    filename = "test_scripts/hparam_test_{}.sbatch".format(counter)
                    counter += 1
                    with open(filename, "w") as templated_file:
                      for line in sbatch_template:
                        if "{{JOB_NAME}}" in line:
                          templated_file.write("#SBATCH --job-name="+job_name+"\n") 
                        elif "{{CMD}}" in line:
                          exec_command = "python learn_nav.py \
--job_id $SLURM_JOBID \
--config_file {CONFIG_FILE} \
--distance {DESCRIPTION} \
--bot_motion {BOT_MOTION} --door_type {DOOR_TYPE} \
--eef_type {EEF_TYPE} \
--n_steps {NSTEPS} --horizon {HORIZON} \
--opt_epochs {OPT_EPOCHS} \
--minibatches {MINIBATCHES} \
--ent_coef {ENT_COEF} \
--vf_coef {VF_COEF} \
--clip_range {CLIP_RANGE} \
--gamma {GAMMA} \
--lam {LAM} \
--rcoef_self_con {RCOEF_SELF_CON} \
--rcoef_force {RCOEF_FORCE} \
--reset_on_large_force {RESET_ON_LARGE_FORCE} \
--rcoef_dist_to_door {RCOEF_DIST_TO_DOOR} \
--robot_theta {ROBOT_THETA} \
--robot_pos 0 0 0 \
--door_init_qpos {DOOR_THETA} \
--goal_offset {GOAL_OFFSET}".format(\
                                CONFIG_FILE = CONFIG_FILE,\
                                DESCRIPTION = DESCRIPTION,\
                                BOT_MOTION = BOT_MOTION,\
                                DOOR_TYPE = DOOR_TYPE,\
                                EEF_TYPE = EEF_TYPE,\
                                NSTEPS = NSTEPS,\
                                HORIZON              = HORIZON,\
                                OPT_EPOCHS = OPT_EPOCHS,\
                                MINIBATCHES = MINIBATCHES,\
                                ENT_COEF             = ENT_COEF,\
                                VF_COEF              = VF_COEF,\
                                CLIP_RANGE           = CLIP_RANGE,\
                                GAMMA                = GAMMA,\
                                LAM                  = LAM,\
                                RCOEF_SELF_CON       = RCOEF_SELF_CON,\
                                RCOEF_FORCE          = RCOEF_FORCE,\
                                RCOEF_DIST_TO_DOOR   = RCOEF_DIST_TO_DOOR,\
                                RESET_ON_LARGE_FORCE = RESET_ON_LARGE_FORCE,\
                                ROBOT_THETA          = ROBOT_THETA,\
                                DOOR_THETA           = DOOR_THETA,\
                                GOAL_OFFSET          = GOAL_OFFSET\
                                )
                          print("Executing command: {}".format(exec_command))
                          templated_file.write(exec_command)
                        else:
                          templated_file.write(line)

                    cmd = "sbatch " + filename
                    os.system(cmd) 

if __name__ == '__main__':
  main()
