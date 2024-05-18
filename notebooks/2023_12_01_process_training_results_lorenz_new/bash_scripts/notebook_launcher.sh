#!/bin/bash

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT="$PATH_PARENT/../../.."
CONDA_ENV=$(cat "$PATH_CONTENT_ROOT"/bash_scripts/conda_env_name.txt)
export PYTHONPATH="${PYTHONPATH}:$PATH_CONTENT_ROOT"

# Add command line arguments to provide the name of the notebook with the extension
while getopts 'n:' flag; do
  case "${flag}" in
  n) NAME_NOTEBOOK="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# If the name of the notebook was not provided, raise an error
if [ -z "$NAME_NOTEBOOK" ]; then
  echo Missing notebook name -n option.
  exit
fi


NAME_OUTPUT_DIR=$( basename "$( cd "$PATH_PARENT/.." || exit; pwd -P )")

PATH_NOTEBOOK="$PATH_PARENT"/../"$NAME_NOTEBOOK"

PATH_OUTPUT_DIR="$PATH_CONTENT_ROOT"/data/notebooks/"$NAME_OUTPUT_DIR"
ARRAY_CONFIG_FILES=("${PATH_PARENT}"/../configs/*.yaml)

echo "PATH_NOTEBOOK: $PATH_NOTEBOOK"
echo "PATH_OUTPUT_DIR: $PATH_OUTPUT_DIR"

for path_yaml_file in "${ARRAY_CONFIG_FILES[@]}"; do
    echo Following config is procesessed: "$file_name"
    export PATH_YAML_CONFIG="$path_yaml_file"
    basename_yaml_file=$(basename "$path_yaml_file" .yaml)
    echo "basename_yaml_file: $basename_yaml_file"
#     conda run --no-capture-output -n "$CONDA_ENV" jupyter nbconvert --execute --to html "$PATH_NOTEBOOK" --output-dir="$PATH_OUTPUT_DIR" --output "${path_yaml_file%.*}".html
    conda run --no-capture-output -n "$CONDA_ENV" jupyter nbconvert --execute --to html "$PATH_NOTEBOOK" --output-dir="$PATH_OUTPUT_DIR" --output "$basename_yaml_file".html
done
