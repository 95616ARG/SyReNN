#!/bin/bash

IMAGE_NAME=masotoud_syrenn

running_containers=$(docker container ls -f "ancestor=$IMAGE_NAME" -f "status=running" -q)
n_containers=$(echo "$running_containers" | wc -l)

if [ -z $running_containers ]; then
    echo "Starting new Docker container..."
    docker run --rm -i -t -v /sys/fs/cgroup:/sys/fs/cgroup:rw -p 50051:50051 \
        -v $PWD:/iovol:rw -w /iovol $IMAGE_NAME \
        "${@:1}"
elif [ $n_containers = 1 ]; then
    container_name=$(echo $running_containers | head -n 1)
    echo "Attaching to existing Docker container..."
    docker exec -i -t -w /iovol $container_name "${@:1}"
else
    echo "Too many containers from the image $IMAGE_NAME, please kill all but one."
fi
