#!/usr/bin/env bash

IMG_NAME="lsun_dataset/sdk:0.0.1"

docker build --tag "$IMG_NAME" lsun_docker \
  -f lsun_docker/Dockerfile
