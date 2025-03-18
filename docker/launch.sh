#!/usr/bin/env bash

IMG_NAME="ctm/sdk:0.0.1"
CONTAINER_NAME="ctm_docker"

docker_run_user () {
  tempdir=$(mktemp -d)
  getent passwd > ${tempdir}/passwd
  getent group > ${tempdir}/group
  docker run -v${HOME}:${HOME} -w$(pwd) --rm -u$(id -u):$(id -g\
) $(for i in $(id -G); do echo -n ' --group-add='$i; done) -v ${tempdir}/passwd:/etc/passwd:ro -v ${tempdir}/group:/etc/group:ro -v /etc/localtime:/etc/localtime:ro -v /dev/shm:/dev/shm "$@"
}


docker_run_user --name $CONTAINER_NAME -it --rm \
    --net=host \
    --ipc=host \
    --gpus all \
    --entrypoint /bin/bash \
    -v "$PWD":/code \
    -v "/data2":/data2 \
    "$IMG_NAME"
