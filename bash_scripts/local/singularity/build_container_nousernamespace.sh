PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

NAME_SINGULARITY_DEFINITION_FILE="lbmpc_semimarkov_nousernamespace.def"
PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/definition_files
PATH_SINGULARITY_DEFINITION_FILE="$PATH_SINGULARITY_DEFINITION_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

# User and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

NAME_IMAGE_SIF_FILE="hydrogym-firedrake_nousernamespace_uid-${USER_ID}_gid-${GROUP_ID}_hostname-$(hostname).sif"
PATH_SIF_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/images
PATH_SIF_FILE="$PATH_SIF_FILE_DIR"/"$NAME_IMAGE_SIF_FILE"

# Build the Singularity image
singularity build \
--no-cleanup \
--fakeroot \
"$(readlink -f "$PATH_SIF_FILE")" \
"$(readlink -f "$PATH_SINGULARITY_DEFINITION_FILE")"
