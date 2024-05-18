#!/bin/bash

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )


# OLD1="init_cond_sinus_frequency_epsilon: 0.0"


while getopts 'f:n:' flag; do
  case "${flag}" in
    f) PATH_FILE="${OPTARG}" ;;
    n) FOLDER_NAME="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

if [ -z "$PATH_FILE" ] || [ -z "$FOLDER_NAME" ]; then
  echo Missing option.
  exit
fi


PATH_OUTPUT_DIR="$PATH_PARENT/../../configs/batch/$FOLDER_NAME"
mkdir -p "$PATH_OUTPUT_DIR"

echo Directory "$PATH_OUTPUT_DIR" as been created.

echo Start creating config files.

# shellcheck disable=SC2207
# Values to replace in config
ARRAY=($(seq 0 5))
# Beginning of the suffix of each of the configuration files .yaml
PRESUFFIX="seed"

for value in "${ARRAY[@]}"; do
  export value
  echo Following value to be inserted: "$value"
  SUBSTITUTE="\$value"
  SUFFIX="$PRESUFFIX"_"${PATH_FILE##*/}"_"$value"
#  sed -f "$PATH_PARENT"/sed_scripts/script.sed -e "s/$OLD1/init_cond_sinus_frequency_epsilon:  $i/" "$PATH_FILE" > "$PATH_OUTPUT_DIR"/"$FOLDER_NAME"_"$i".yaml
#  sed -f <(i="$i" envsubst "$i" <"$PATH_PARENT"/sed_scripts/script.sed) "$PATH_FILE" > "$PATH_OUTPUT_DIR"/"$FOLDER_NAME"_"$i".yaml
#  sed -f <(envsubst '$i' <"$PATH_PARENT"/sed_scripts/script.sed) "$PATH_FILE" > "$PATH_OUTPUT_DIR"/"$FOLDER_NAME"_"$i".yaml
  sed -f <(envsubst $SUBSTITUTE <"$PATH_PARENT"/sed_scripts/script.sed) "$PATH_FILE" > "$PATH_OUTPUT_DIR"/"$FOLDER_NAME"_"$SUFFIX".yaml
done
