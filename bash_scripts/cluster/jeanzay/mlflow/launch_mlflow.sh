#!/bin/bash

# Path of the parent directory of this script
PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
# Path of the root of the project
PATH_CONTENT_ROOT="$PATH_PARENT"/../../../..

V_ENV_NAME="venvbarl"
PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate

# Define the port to use for the MLFlow UI
PORT=5001

# Path of the "store", i.e., where the mlflow runs are stored
PATH_BACKEND_STORE="$PATH_CONTENT_ROOT"/experiments/mlruns

echo "Loading virtual environment"
echo

# Load virtual environment
source "$PATH_VENV"  && echo "Activation of virtual environment: $V_ENV_NAME"

echo PATH_VENV: "$PATH_VENV"
echo
echo PATH_CONTENT_ROOT: "$PATH_CONTENT_ROOT"
echo
echo "Chosen backend store: $PATH_BACKEND_STORE"
echo
echo "PORT: $PORT"
echo
echo "Starting MLFlow UI"
echo

# Run mlflow
mlflow ui --backend-store-uri "$PATH_BACKEND_STORE" --port "$PORT" --host 0.0.0.0

#  To join the server: https://jupyterhub.idris.fr/user/ucd32aq/jupyter_4/proxy/5001/
#