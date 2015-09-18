#!/bin/bash
DOCKER_IMG="sean/dl"
DEMO_CONTAINER_NAME="dl-service"
sudo docker run -it \
    -v $(pwd)/flask-server:/opt/flask-server \
    -v $(pwd)/files:/opt/files \
    -v $(pwd)/data:/opt/data \
    -p 8889:8888 \
    --name $DEMO_CONTAINER_NAME \
    $DOCKER_IMG bash