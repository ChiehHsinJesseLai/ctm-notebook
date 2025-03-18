#!/usr/bin/env bash

IMG_NAME="lsun_dataset/sdk:0.0.1"
CONTAINER_NAME="lsun_dataset"

docker_run_user () {
  tempdir=$(mktemp -d)
  getent passwd > ${tempdir}/passwd
  getent group > ${tempdir}/group
  docker run -v${HOME}:${HOME} -w$(pwd) --rm -u$(id -u):$(id -g\
) $(for i in $(id -G); do echo -n ' --group-add='$i; done) -v ${tempdir}/passwd:/etc/passwd:ro -v ${tempdir}/group:/etc/group:ro "$@"
}


docker_run_user --name $CONTAINER_NAME -it --rm \
    --net=host \
    --ipc=host \
    --gpus all \
    --entrypoint /bin/bash \
    -v "$PWD":/code \
    -v "/dataset":/dataset \
    "$IMG_NAME"
