#!/bin/bash

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

NAME_MOUNT_DIR="mount_dir"
PATH_CONTAINER_CONTENT_ROOT="/home/firedrake/$NAME_MOUNT_DIR/project_root"

NAME_CONTAINER="lbmpc_semimarkov"

# Run a shell in the container
docker run \
  -it \
  --rm \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/lbmpc_semimarkov,target="$PATH_CONTAINER_CONTENT_ROOT"/lbmpc_semimarkov \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/data,target="$PATH_CONTAINER_CONTENT_ROOT"/data \
  "$NAME_CONTAINER"

  # --read-only: Mount the container's root filesystem as read-only to check compatibility with Singularity
  # Indeed, Singularity containers are read-only by default
  # Unfortunately, firedrake writes at least in /home/firedrake/.cache/pytools in the container,
  # so we cannot use --read-only. Consequently, the decision is to make the Singularity container writable
