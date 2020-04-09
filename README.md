# SyReNN: Symbolic Representations for Neural Networks
SyReNN (pronounced Siren) is a library for analyzing deep neural networks by
_enumerating the linear regions_ of piecewise-linear (eg.  ReLU) functions on
one- and two-dimensional input "restriction domains of interest."

In particular, this repository contains the code described and utilized in the
papers:

**"Computing Linear Restrictions of Neural Networks" ([1])**

[Conference on Neural Information Processing Systems (NeurIPS)
2019](https://neurips.cc/Conferences/2019)

Links:
[Paper](https://papers.nips.cc/paper/9562-computing-linear-restrictions-of-neural-networks),
[Slides](https://zenodo.org/record/3520104),
[Poster](https://zenodo.org/record/3520102)
```
@incollection{sotoudeh:linear_restrictions,
  title = {Computing Linear Restrictions of Neural Networks},
  author = {Sotoudeh, Matthew and Thakur, Aditya V},
  booktitle = {Advances in Neural Information Processing Systems 32},
  editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
  pages = {14132--14143},
  year = {2019},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/9562-computing-linear-restrictions-of-neural-networks.pdf}
}
```

**"A Symbolic Neural Network Representation and its Application to
Understanding, Verifying, and Patching Networks" ([2])**

Links: [Preprint](https://arxiv.org/abs/1908.06223)
```
@article{sotoudeh:symbolic_networks,
  author    = {Matthew Sotoudeh and Aditya V. Thakur},
  title     = {A Symbolic Neural Network Representation and its Application to
  Understanding, Verifying, and Patching Networks},
  journal   = {CoRR},
  volume    = {abs/1908.06223},
  year      = {2019},
  url       = {https://arxiv.org/abs/1908.06223},
  archivePrefix = {arXiv},
  eprint    = {1908.06223},
}
```

**"Correcting Deep Neural Networks with Small, Generalizing Patches ([3])**

[NeurIPS 2019 Workshop on Safety and Robustness in Decision
Making](https://sites.google.com/view/neurips19-safe-robust-workshop)

Links: [Paper](https://drive.google.com/file/d/0B3mY6u_lryzdNTFaZnkzUzhuRDNnZG9rdDV5aDA2WFpBOWhN/view)
```
@inproceedings{sotoudeh:correcting_dnns_srdm19,
  title={Correcting Deep Neural Networks with Small, Generalizing Patches},
  author={Sotoudeh, Matthew and Thakur, Aditya V},
  booktitle={NeurIPS 2019 Workshop on Safety and Robustness in Decision Making},
  year={2019}
}
```

We will refer to these as ``[1]``, ``[2]``, and ``[3]`` respectively in
comments and code.

## Notation and Naming
In the papers, we often use mathematical notation that can be hard/impossible
to use in ASCII-encoded code. Because of this, we use slightly different
terminology in this repository than in the papers. Here is a rough translation
dictionary

- SyReNN refers to both this library and the symbolic representation that we
  compute (i.e., the linear partitions). In [1], this is referred to (when
  restricted to one-dimensional inputs) as "ExactLine." In [2], this is
  referred to as "the symbolic representation f-hat."
- "ExactLine" is often used to refer to the SyReNN symbolic representation when
  the input restriction domain of interest is a one-dimensional line segment.
  It corresponds to ExactLine and curly-P in [1].
- "Transforming" a line or polytope under a given network or layer involves
  computing SyReNN for that network or layer. It corresponds to the Extend and
  circle-times operators discussed in [1] and [2].

## Quickstart
### Prerequisites
We provide two ways to build the code, either locally or in a Docker container.
See each sub-section below for the prerequisites.

We recommend at least 16 GB of memory and 8 physical processor cores for
running the library, although smaller machines should be able to run it. The
reference environment is an AWS EC2 ``c5.metal`` instance using ``runexec`` to
limit use to only the first 16 cores as shown in ``.bazelrc``.

Note that because we compile (almost) everything from source, the first build
*will likely take a long time*; at least a few minutes. Subsequent builds
should be much faster. If you have multiple processing cores on your machine
and would like to use them to speed up the build, you can modify the ``make``
commands in ``external/*.BUILD`` to ``make -j`` followed by the number of
threads you would like Make to run. However, because Bazel will simultaneously
be running other builds processes, too much use of parallelism in the ``make``
calls can sometimes cause freezing and running out of memory. If your computer
is already freezing/running out of memory, consider removing the existing
``-j#`` flags from ``external/*.BUILD``.

#### Local Builds
You must install [Bazel](https://bazel.build/) 1.1.0 and have binaries for
building arbitrary C++ packages (eg. ``build-essential`` for Ubuntu).
Furthermore, the ``libcairo2``, ``libffi-dev``, ``zlib1g-dev``, ``zip``, and
``libgmp3-dev`` packages are required for the Python code (but usually come
pre-installed with most desktop Linux distributions). A
[Dockerfile](Dockerfile) is provided with a compatible setup for reference.
Note that Bazel will _automatically_ download and build copies of OpenSSL,
Python 3.7.4, Intel TBB, Intel MKLDNN, Eigen, OpenBLAS, PyTorch, and all PIP
dependencies to the Bazel environment when applicable --- they do not need to
be manually installed.

#### Docker Builds
Alternatively, a Docker container is provided to simplify the build and running
process. To use it, first build the image with ``./docker_build.sh`` then
prepend ``./docker_run.sh`` to all of the commands below. For example, instead
of ``make start_server``, use ``./docker_run.sh make start_server``. Everything
should be handled transparently. **NOTE:** Benchexec is currently not supported
under the Docker container due to permission errors. It is possible to resolve
this by removing all of the user-related lines from [Dockerfile](Dockerfile),
but that will likely cause issues with Bazel and generated file permissions.

### Configuration
There are two major things to configure before using the code:

1. In ``syrenn_server/server.cc``, modify the ``consexpr size_t
   memory_usage = ...;`` threshold to suit the desired memory usage. Please
   note that this is a rough upper-bound on memory usage per-call by the server
   and not exact. A reasonable default is half of the RAM available on your
   machine.
2. (Optional) If you have ``benchexec`` installed and would like run under
   ``runexec``, the last line in ``.bazelrc`` can be uncommented and updated
   with the desired parameters. **WARNING:** ``benchexec`` does not seem to be
   correctly flushing terminal input to the process, so if you use this you
   will need to pipe the input in (eg. ``echo -e "1\n" | make
   integrated_gradients_experiment``). **WARNING:** The rules in
   [third_party/](third_party) use ``benchexec`` within their Docker
   containers, so please comment out the ``runexec`` line in ``.bazelrc``
   before running any of the rules in [third_party/](third_party).

### Running our Experiments
If you are only interested in reproducing the experiments from our papers, they
are all included in the [experiments/](experiments/) directory and can be run
using Bazel:

1. Install the prerequisites.
2. Clone this repository and run ``make start_server`` to start the C++ server.
3. In a different terminal, navigate back to this repository and run ``make
   {experiment}_experiment`` where {experiment} is the name of the experiment
   you wish to run. See [Makefile](Makefile) for a list of supported
   experiments. ``make all_experiments`` will run all experiments, but be
   warned this may take some time.

After running an experiment, results should appear in the directory
``experiments/results``  as a ``.exp.tgz`` archive (currently-running,
incomplete, or errored experiments may store partial results in a sub-directory
of ``experiments/results``). See [experiments/README.md](experiments/README.md)
for more information about the experiments.

Portions of the experiments from [2] use prior work
[ReluPlex](https://github.com/guykatzz/ReluplexCav2017/) and
[ERAN](https://github.com/eth-sri/eran); scripts and documentation for running
those are under ``third_party/``.

### Using in Your Own (Python) Project
SyReNN can also be used as a library ("PySyReNN") in your own Python project.
**Please note that we only test against Python 3.7.4.**

1. Install the prerequisites above.
2. Clone this repository and run ``make start_server`` to start the C++
   server.
3. In a different terminal and in your own project directory (i.e., *not* in
   this repository), install [PyTorch](https://pytorch.org/), then run ``pip3
   install --user pysyrenn`` to install the Python front-end and client
   library.

You should now be able to ``import pysyrenn`` and use the front-end. See
[pysyrenn/frontend/README.md](pysyrenn/frontend/README.md),
[pysyrenn/helpers/README.md](pysyrenn/helpers/README.md), and
[experiments](experiments) for usage examples.

### Running our Tests
We have included a number of correctness tests for the library. All tests can
be run with the command ``bazel test //...``. HTML coverage reports for the
PySyReNN front-end can be generated with ``make pysyrenn_coverage`` (an
``htmlcov/`` directory should be created).

## High-Level Overview
There are three main pieces to the library:

1. [syrenn_server/](syrenn_server) contains C++ implementations of the
   underlying algorithms along with a gRPC server that exposes a thin Protobuf
   front-end to them.
2. [pysyrenn/](pysyrenn) contains the Python client library, which corresponds
   to the ``pysyrenn`` pip package. It is subdivided into two sub-directories:
    1. [pysyrenn/frontend/](pysyrenn/frontend) contains a lightweight Python
       front-end to the C++ server. It provides an abstraction consisting of
       ``Layer``s and a ``Network`` containing a sequence of layers. It
       supports loading a Network from [ERAN](https://github.com/eth-sri/eran)
       and [ONNX](https://github.com/onnx/models) file formats. The symbolic
       representation of a network can then be computed, given a bounded
       restriction domain of interest in V-representation.
    2. [pysyrenn/helpers/](pysyrenn/helpers/) contains a number of helper
       methods that utilize the symbolic representation. Examples include
       computing exact integrated gradients and exactly characterizing decision
       boundaries.  These strive to be relatively opaque with respect to the
       symbolic representation, eg., you can use the integrated gradients
       helper to exactly compute integrated gradients without ever worrying
       about the symbolic representation.
5. [experiments/](experiments) contains Python scripts which execute the
   experiments from our published work. These are particularly useful as
   examples of using the client library ``pysyrenn``.

A few other directories also exist:

- [external/](external) contains Bazel BUILD rules for GTEST, Python 3.7.4 (including
  OpenSSL), and TBB.
- [models/](models) contains Bazel rules either generating, downloading, or
  referencing trained models from the ERAN, ACAS Xu, and VRL projects (prior
  work). ERAN models are downloaded in WORKSPACE and referenced in
  ``models/BUILD``. ACAS Xu models are downloaded and converted to the ERAN
  format by the Bazel genrule ``translate-acas-models`` and Python script
  ``translate_acas_model.py``. SyReNN-compatible VRL models are included in
  this repository under ``vrl/eran/*.eran`` along with the code used to
  generate them from the official VRL repository.
- [third_party/](third_party) contains Docker images and Bazel rules for
  running the experiments from our paper that use ERAN and ReluPlex.
- [pip_info/](pip_info) contains metadata for the PyPI package ``pysyrenn``.

## Note on Floating Point Arithmetic and Reproducibility
As mentioned in our papers, we utilize floating point arithmetic throughout the
codebase. This can cause small errors in the computation of vertex positions,
meaning that the symbolic representation produced by the server may not be
_perfectly_ accurate (although, in practice, it is quite accurate --- see our
papers for examples).

Furthermore, as the MKL-DNN library used for efficient convolutions utilizes
different algorithms and thus produces different results on different machines,
exact numerical reproducibility is not guaranteed when executing on different
machines. However, all reported results should be reproducible within a
reasonable tolerance. If you are unable to reproduce some results, please
contact us.

Finally, we note that many models have "non-linear hotspots" where floating
point issues become particularly important, usually when the restriction domain
of interest includes the origin (which causes most activations throughout the
network to hover about 0, right on the edge between linear regions). For both
numerical precision and performance, when possible, we recommend modifying
queries to the transformer server to avoid the origin. An example of this is in
[experiments/model_checking.py](experiments/model_checking.py) from [2], where
we can a-priori guarantee a particular region about the origin maps back into
the initial space, and thus avoid computing the symbolic representation in that
region.

## Usage & Documentation of Python Library
Please see [pysyrenn/frontend/README.md](pysyrenn/frontend/README.md) and
[pysyrenn/helpers/README.md](pysyrenn/helpers/README.md) for documentation. The
best usage examples are in [experiments/](experiments/), however there are also
useful examples in [pysyrenn/frontend/tests](pysyrenn/frontend/tests) and
[pysyrenn/helpers/tests](pysyrenn/helpers/tests).

Information about supported layers can be found in
[syrenn_server/LayerSupport.md](syrenn_server/LayerSupport.md)

## People
- [Aditya Thakur](https://thakur.cs.ucdavis.edu/) can be reached at
  [avthakur@ucdavis.edu](mailto:avthakur@ucdavis.edu).
- [Matthew Sotoudeh](https://matthewsot.github.io/) can be reached at
  [masotoudeh@ucdavis.edu](mailto:masotoudeh@ucdavis.edu).
