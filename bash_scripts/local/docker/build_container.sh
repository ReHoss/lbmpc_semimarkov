# Description: Build a container with the current user's UID and GID

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")
PATH_DOCKERFILE_DIR="$PATH_CONTENT_ROOT"/docker
TAG_IMAGE="lbmpc_semimarkov"


echo "Building the Docker image from the Dockerfile in $PATH_DOCKERFILE_DIR"
echo "The image will be tagged as $TAG_IMAGE"
echo "The current user's UID and GID will be passed to the Dockerfile"
echo

echo "Command:
  docker build \
  --tag $TAG_IMAGE \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --progress=plain \
  $PATH_DOCKERFILE_DIR"

docker build \
  --tag "$TAG_IMAGE" \
  --build-arg USER_ID="$(id -u)" \
  --build-arg GROUP_ID="$(id -g)" \
  --progress=plain \
  "$PATH_DOCKERFILE_DIR"



# --tag hydrogym-firedrake: Name of the image
# --build-arg USER_ID=$(id -u): Pass the current user's UID to the Dockerfile
# --build-arg GROUP_ID=$(id -g): Pass the current user's GID to the Dockerfile
# --progress=plain: Show the build progress in a plain format (add verbosity)
# "$PATH_DOCKERFILE_DIR": Path to the directory containing the Dockerfile to build the image from