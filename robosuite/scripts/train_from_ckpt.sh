#!/bin/bash
DIR=$1
CF=$DIR"/args.json"
MODEL=$DIR"/366080model.ckpt"
#MODEL=$DIR"/4326400model.ckpt"
#MODEL=$DIR"/1331200model.ckpt"
python learn_door.py --model $MODEL --config_file $CF
