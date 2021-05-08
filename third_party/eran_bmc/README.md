# ERAN-BMC
This directory includes a Dockerfile and code to demonstrate imprecision when
using the [ERAN project](https://github.com/eth-sri/eran) for Bounded Model
Checking.

We currently only test the [DeepPoly](https://doi.org/10.1145/3290354) domain
on the Pendulum model. Other domains may be implemented in the future.

# Building & Running
1. Install Bazel 4.0.0, Docker (tested with version 18.09.6, build 481bc77),
   and any prerequisites mentioned in [../../README.md](../../README.md).
2. Ensure that the ``runexec`` line in [../../.bazelrc](../../.bazelrc) is
   commented out (we run ``runexec`` directly in the Docker containers).
3. From the root of this repository, run ``bazel run
   third_party/eran_bmc:experiment``. This target will build the Docker
   container then run the experiment; you should see the output ``ERAN reported
   initLB unsafe after the first step.``, which indicates ERAN found a spurious
   counterexample before it could verify any steps.
