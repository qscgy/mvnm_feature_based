#!/usr/bin/env bash

workdir=`pwd`
prog="$workdir/$1"
containerprog="/work/mvnm/$1"

# && chmod 777 cosmicrays/plot.png && chmod 777 cosmicrays/galactic_trajectories.txt
docker run -it -v "$workdir:/work/mvnm/" --user $(id -u):$(id -g) convolve/mvnm bash -c "cd mvnm; python $containerprog $2"
#docker run -it -v "$workdir:/cosmicrays" --entrypoint="/bin/bash" convolve/crpropa