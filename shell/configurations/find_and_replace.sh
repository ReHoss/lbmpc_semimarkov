#!/bin/bash

# This script replaces values from a bunch of configs, nice for decreasing xp time.

#PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )

while getopts 'f:n:' flag; do
  case "${flag}" in
#    f) PATH_FILE="${OPTARG}" ;;
    n) PATH_FOLDER_CONFIGS="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ -z "$PATH_FOLDER_CONFIGS" ]; then
  echo Missing option.
  exit
fi

# Create a folder to store the modified configs with date and time.
NAME_FOLDER_NEW_CONFIGS=test_transfer_learning
PATH_FOLDER_NEW_CONFIGS="$PATH_FOLDER_CONFIGS"/../"$NAME_FOLDER_NEW_CONFIGS"_$(date +%Y_%m_%d_%H_%M_%S)
mkdir "$PATH_FOLDER_NEW_CONFIGS"
echo "New folder created: $PATH_FOLDER_NEW_CONFIGS"

for file in "$PATH_FOLDER_CONFIGS"/*; do
  echo "Processing: "
# Remark: $file contains the absolute path !
  echo "$file"
  echo
  sed -e 's/total_timesteps: 2000000/total_timesteps: 4000/' -e 's/decay_horizon: 200000/decay_horizon: 2000/' -e 's/eval_freq: 20000/eval_freq: 2000/' "$file" > "$PATH_FOLDER_NEW_CONFIGS"/"${file##*/}"
  echo
done


# --- Examples --- :

#find $PATH_FOLDER_CONFIGS -type f -name '*' -exec sed -e 's/pi:/- pi:/' -e 's/qf:/- vf:/' -e 's/tqc/ppo/' {} \;

#sed -e 's/deterministic: True/deterministic: False/' -e 's/eval_freq: 2000/eval_freq: 10000/' "$file"
#sed -i -e 's/total_timesteps: 1000/total_timesteps: 100000/' "$file"
#sed -e 's/pi:/- pi:/' -e 's/qf:/- vf:/' -e 's/tqc/ppo/' "$file" > "$file"_ppo.yaml
#sed '/dtype: float32/a \ \ \ \ dict_scaling_constants:\n      observation: 8\n      state: 4' "$file" > "$PATH_FOLDER_CONFIGS"/../overfitting_new3/"${file##*/}"
#sed '/dtype: float32/a \ \ \ \ dict_scaling_constants:\n      obervation: 8\n      state: 4'  /home/hosseinkhan/Documents/work/phd/drloc-sb3/configs/batch/overfitting_new2/*
#sed '/dtype: float32/a \ \ \ \ dict_scaling_constants:\n      observation: 8\n      state: 4' "$file" > "$PATH_FOLDER_CONFIGS"/../overfitting_new3/"${file##*/}"
#sed -e 's/entropy_decay_callback: {}/entropy_decay_callback:\n    dict_decay:\n      decay_horizon: 200000\n/' "$file"


