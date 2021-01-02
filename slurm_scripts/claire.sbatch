#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=napoli-cpu
#################
#number of GPUs
##SBATCH --gres=cpu:0
#################
#set a job name  
#{{JOB_NAME}}
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/sailhome/clairech/robosuite/slurm_jobs/%j.out
#################
# a file for errors from the job
#SBATCH --error=/sailhome/clairech/robosuite/slurm_jobs/%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=18:00:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,  
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition 
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM) 
##SBATCH -p dev 
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=12000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 4
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=clairech@stanford.edu
# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
#now run normal batch commands
# Begins in slurm_scripts
BOT_MOTION="mmp"
DOOR_TYPE="dpnlrw"
EEF_TYPE="static"
DOOR_THETA=2.0
ROBOT_THETA=0.0
ROBOT_POS=(0 1.0 0)
GOAL_OFFSET=(2.5 0)

NSTEPS=8192
HORIZON=4096
OPT_EPOCHS=10
MINIBATCHES=4
ENT_COEF=0.0
VF_COEF=0.5
CLIP_RANGE=0.2
GAMMA=0.99
LAM=0.95

RCOEF_DIST_TO_HANDLE=0
RCOEF_DOOR_ANGLE=0
RCOEF_HANDLE_CON=0
RCOEF_BODY_DOOR_CON=-1
RCOEF_SELF_CON=0
RCOEF_ARM_HANDLE_CON=0
RCOEF_ARM_DOOR_CON=0
RCOEF_FORCE=-10
RCOEF_GRIPPER_TOUCH=0
RCOEF_DIST_TO_DOOR=1
RCOEF_WALL_CON=-1

RESET_ON_LARGE_FORCE="t"

cd /sailhome/clairech/robosuite/robosuite/scripts

# train new learn_door.py
#python learn_door.py \
python learn_nav.py \
  --job_id $SLURM_JOBID \
  --config_file ~/robosuite/robosuite/scripts/default_door.json \
  --bot_motion $BOT_MOTION --door_type $DOOR_TYPE \
  --eef_type $EEF_TYPE \
  --n_steps $NSTEPS --horizon $HORIZON \
  --opt_epochs $OPT_EPOCHS \
  --minibatches $MINIBATCHES \
  --ent_coef $ENT_COEF \
  --vf_coef $VF_COEF \
  --clip_range $CLIP_RANGE \
  --gamma $GAMMA \
  --lam $LAM \
  --rcoef_dist_to_handle $RCOEF_DIST_TO_HANDLE \
  --rcoef_door_angle $RCOEF_DOOR_ANGLE \
  --rcoef_handle_con $RCOEF_HANDLE_CON \
  --rcoef_body_door_con $RCOEF_BODY_DOOR_CON \
  --rcoef_self_con $RCOEF_SELF_CON \
  --rcoef_arm_handle_con $RCOEF_ARM_HANDLE_CON \
  --rcoef_arm_door_con $RCOEF_ARM_DOOR_CON \
  --rcoef_force $RCOEF_FORCE \
  --reset_on_large_force $RESET_ON_LARGE_FORCE \
  --rcoef_gripper_touch $RCOEF_GRIPPER_TOUCH \
  --rcoef_dist_to_door $RCOEF_DIST_TO_DOOR \
  --rcoef_wall_con $RCOEF_WALL_CON \
  --robot_theta $ROBOT_THETA \
  --robot_pos ${ROBOT_POS[*]} \
  --door_init_qpos $DOOR_THETA \
  --goal_offset ${GOAL_OFFSET[*]}

# train from checkpoint
#DIR="/sailhome/clairech/robosuite/learned_models/wide_670099_cr.0.2_diq.2.0_dt.dpnlrw_et.static_ec.0.0_g.0.99_go.0.0.1.2_h.4096_l.0.95_m.4_nc.4_ns.8192_oe.10_radc.0.0_rahc.0.0_rbdc.-1.0_rdtd.1.0_rdth.0.0_rda.0.0_rf.-10.0_rgt.0.0_rhc.0.0_rsc.0.0_rwc.-1.0_rolf.True_ra.ppo2_rp.-1.0.-1.5.0.0_rt.1.5_vc.0.5"
#CF=$DIR"/args.json"
#MODEL=$DIR"/4985600model.ckpt"
#python learn_door.py --model $MODEL --config_file $CF


# done
echo "Done"
exit 0

