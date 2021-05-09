# ReluPlex-BMC
Code to use ReluPlex for bounded-model-checking. This is the ReluPlex
counter-part to
[../../experiments/model_checking.py](../../experiments/model_checking.py).

# Building & Running
1. Install Bazel 4.0.0, Docker (tested with version 18.09.6, build 481bc77),
   and any prerequisites mentioned in [../../README.md](../../README.md).
2. Ensure that the ``runexec`` line in [../../.bazelrc](../../.bazelrc) is
   commented out (we run ``runexec`` directly in the Docker containers).
3. From the root of this repository, run ``bazel run
   third_party/reluplex_model_checking:experiment``. This target will build the
   Docker container then run the experiment; once finished, results will be
   copied into a ``.tgz`` file in this directory.

# Notes
## Timeouts
The duration of the experiment can be controlled in ``experiment.sh`` by
changing the variable ``sh_timeout`` in ``run_for_model``. Please note that
there is some overhead, so if you wish to see how many steps it can verify in,
say, an hour, you should set the timeout to something closer to 1.5 hours.

## Results
Results will be written to a ``.tgz`` file in this directory with one CSV for
each of the three models.

The CSV needs significant pre-processing before its results can be compared to
that of the paper:

1. Verification for each step is broken into 16 different specs, because both
   our unsafe region and initial region are non-convex polytopes (each is,
   however, a union of 4 convex polytopes). Therefore, the times from each of
   the 16 specs per step should be added together to get the overall time of
   each step.
2. Each step must be verified individually with ReluPlex, so the time needed to
   verify that _all steps up to i_ are safe, you need the cumulative time of
   each individual step up to i. Thus, cumulative time should be taken by
   essentially "adding down the columns" of the generated CSV.

The basic takeaways are: (1) 16 rows in the CSV -> 1 timestep and (2) you need
to take the cumulative sum down the time column.

This script cannot be safely early-quitted, instead please use the
``sh_timeout`` variable as described above.
