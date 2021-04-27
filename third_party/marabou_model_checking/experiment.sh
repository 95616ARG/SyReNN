#!/bin/bash

image_file=$PWD/third_party/marabou_model_checking/marabou_bmc.tgz
specs_tgz=$PWD/third_party/reluplex_model_checking/specs.tgz
builddir=$BUILD_WORKING_DIRECTORY
outdir=$builddir/third_party/marabou_model_checking
local_outdir=$PWD/local_results

docker load -i $image_file
# First, extract all of the specs (we assume their names).
tar -xzf $specs_tgz
models="pendulum_continuous satelite quadcopter"

function run_for_model {
    model=$1
    # This is the amount of overall-wallclock time (in seconds) to spend per
    # spec.  NOTE: This should be greater than the desired experiment timeout,
    # as there is overhead in starting the scripts.
    sh_timeout=$((90*60))
    # https://stackoverflow.com/questions/17066250
    SECONDS=0
    net_file="$model/stepnet.nnet"
    result_csv="$local_outdir/$model.csv"
    mkdir -p $(dirname $result_csv)
    echo "Model,Spec,Steps,Result,NON-CUMULATIVE Time" > $result_csv
    for steps in {1..100}
    do
        echo "Verifying step: $steps"
        for i in {0..3}
        do
            for j in {0..3}
            do
                echo "NONE" > $local_outdir/bmc_output
                spec_file="$model/0"$i"_0"$j"_spec.nspec"
                echo "Running for spec: $spec_file"
                # Docker file-permissions get a bit ugly, so we want to have a
                # clean separation between input/output directories.
                docker run --rm -t -i \
                    -v $PWD:/ivol:ro \
                    -v $local_outdir:/ovol:rw \
                    -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
                    -w /marabou marabou_bmc \
                    runexec --no-container --walltimelimit 172800 --cores 0-15 --memlimit 17179869184 --output /dev/stdout --input - -- \
                    ./cpp_interface_example/bmc.elf \
                    /ivol/$model/stepnet.nnet /ivol/$spec_file $steps /ovol/bmc_output \
                    > /dev/null
                cat $local_outdir/bmc_output >> $result_csv
                if [ "$SECONDS" -gt "$sh_timeout" ]; then
                    echo "Timeout. Finishing model..."
                    return
                fi
            done
        done
    done
}

rm -rf $local_outdir

for model in $models
do
    run_for_model $model
done

# https://crunchify.com/shell-script-append-timestamp-to-file-name/
filename="$PWD/results_$(date "+%Y.%m.%d-%H.%M.%S").tgz"

# https://stackoverflow.com/questions/939982
cd $local_outdir
tar -zcvf $filename *.csv
rm -rf $local_outdir

# Move the results to the user-visible location.
cp $filename $outdir
