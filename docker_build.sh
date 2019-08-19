#!/bin/bash

IMAGE_NAME=masotoud_syrenn

docker build --force-rm -t $IMAGE_NAME . --build-arg UID=$UID
