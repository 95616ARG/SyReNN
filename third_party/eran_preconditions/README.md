# ERAN-Preconditions
This directory includes a Dockerfile and code for using the [ERAN
project](https://github.com/eth-sri/eran) to compute preconditions on network
outputs (particularly applied to visualization of ACAS Xu decision boundaries).

We currently only support the [DeepPoly](https://doi.org/10.1145/3290354)
domain as it provides the necessary representation for precondition
computation. Other domains may be implemented in the future.

# Building & Running
1. Install Bazel 4.0.0, Docker (tested with version 18.09.6, build 481bc77),
   and any prerequisites mentioned in [../../README.md](../../README.md).
2. Ensure that the ``runexec`` line in [../../.bazelrc](../../.bazelrc) is
   commented out (we run ``runexec`` directly in the Docker containers).
3. From the root of this repository, run ``bazel run
   third_party/eran_preconditions:experiment``. This target will build the
   Docker container then run the experiment; once finished, results will be
   copied into a ``.tgz`` file in this directory.

# Notes
## Expected Time
This experiment may take a long time; on our test machine, the Docker image
usually takes 200-300 seconds for a first build. The underlying computation for
most of the 8 scenarios takes 6 - 7 minutes to complete, while generating the
plots for each one may add an additional 5 - 10 minute overhead.

You can early-quit the script with (**a single**) Ctrl-C, which will write the
outputs generated so far to a tar file.

## Generating More Plots
By default, we only generate plots for the first scenario (but record timing
results for all of them). You can force the script to generate plots for all
scenarios by setting ``do_plot = True`` in ``experiment.py``.

With the default plotting behavior we recommend expecting about 1 hour for
completion, if all plots are made about 2 hours is closer.
