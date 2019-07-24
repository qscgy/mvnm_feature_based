#!/usr/bin/env bash

workdir=`pwd`
prog="$workdir/$1"
containerprog="/mvnm/$1"

# && chmod 777 cosmicrays/plot.png && chmod 777 cosmicrays/galactic_trajectories.txt
docker run -it -v "$workdir:/mvnm" --user $(id -u):$(id -g) convolve/mvnm python "$containerprog" "$2"
#docker run -it -v "$workdir:/cosmicrays" --entrypoint="/bin/bash" convolve/crpropa