PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

NAME_MOUNT_DIR="mount_dir"
PATH_CONTAINER_CONTENT_ROOT="/home/firedrake/$NAME_MOUNT_DIR/project_root"
NAME_CONTAINER="hydrogym-firedrake_nousernamespace_uid-1001_gid-1001_hostname-mecacpt80.sif"
PATH_CONTAINER="$PATH_CONTENT_ROOT"/singularity/images/"$NAME_CONTAINER"

singularity --debug run \
  --no-home \
  --writable-tmpfs \
  --no-init \
  --bind "$PATH_CONTENT_ROOT":"$PATH_CONTAINER_CONTENT_ROOT" \
  "$PATH_CONTAINER"

# --no-mount strings              disable one or more 'mount xxx' options set in singularity.conf, specify absolute destination path to disable a bind path entry, or 'bind-paths' to disable all bind path entries.
# --writable-tmpfs                makes the file system accessible as read-write with non persistent data (with overlay support only)
# --containall                    contain not only file systems, but also PID, IPC, and environment
# --no-init                       do NOT start shim process with --pid
# --no-eval                       do not shell evaluate env vars or OCI container CMD/ENTRYPOINT/ARGS
