#!/bin/bash

builddir=$BUILD_WORKING_DIRECTORY
model="$PWD/models/vrl/eran/pendulum_continuous.eran"
experiment_py="$PWD/third_party/eran_bmc/experiment.py"
image_file="$PWD/third_party/eran_preconditions/eran_image.tgz"

# Copy everything to the local directory and load the Docker image.
cp $model pendulum_continuous.eran
cp $experiment_py experiment.py
docker load -i $image_file

# Run the experiment and save results to local_outdir.
rm -rf local_outdir
mkdir local_outdir

# The Docker container has a habit of messing with __pycache__ permissions in
# /ivol, so we keep /ivol read-only and only give it permission to write to
# /ovol.
docker run --rm -t -i \
    -v $PWD:/ivol:ro \
    -v $PWD/local_outdir:/ovol:rw \
    -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
    -w /eran/tf_verify \
    eran_preconditions \
    runexec --no-container --walltimelimit 172800 --cores 0-15 --memlimit 17179869184 --output /dev/stdout --input - -- \
    python3 /ivol/experiment.py /ivol/pendulum_continuous.eran /ovol
