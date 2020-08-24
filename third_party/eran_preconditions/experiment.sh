#!/bin/bash

builddir=$BUILD_WORKING_DIRECTORY
outdir=$builddir/third_party/eran_preconditions
model="$PWD/models/acas_models/1_1.eran"
polar_image_py="$PWD/experiments/polar_image.py"
experiment_py="$PWD/third_party/eran_preconditions/experiment.py"
image_file="$PWD/third_party/eran_preconditions/eran_image.tgz"

# Copy everything to the local directory and load the Docker image.
cp $model model.eran
cp $polar_image_py polar_image.py
cp $experiment_py experiment.py
docker load -i $image_file

# Remove the last ReLU layer, as we do with SyReNN.
# https://stackoverflow.com/questions
tac model.eran | sed '/ReLU/ {s//Affine/; :loop; n; b loop}' | tac > model.norelu.eran

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
    python3 /ivol/experiment.py /ivol/model.norelu.eran /ovol

# https://crunchify.com/shell-script-append-timestamp-to-file-name/
filename="results_$(date "+%Y.%m.%d-%H.%M.%S").tgz"

# https://stackoverflow.com/questions/939982
cd local_outdir
# NOTE: Doesn't support sub-directories
tar -zcvf ../$filename *
cd ..
rm -rf local_outdir

# Move the results to the user-visible location.
cp $filename $outdir
