# Experiments
These are Python scripts to reproduce experiments from [1] and [2]. Experiments
can be run using the helpers in ``../Makefile`` or by directly calling ``bazel
run experiments:{experiment_name}``. Results are placed in a ``results`` folder
in this directory.

## Experiment Class
Each experiment sub-classes the ``Experiment`` class in ``experiment.py``. This
class is highly opinionated in the assumptions it makes about the experimental
setup.

### Experiment Phases
At a high-level, ``Experiment``s are broken up into two phases:

- The "Run" phase should execute the "data collecting" portion of the
  experiment, but does *not* actually produce any figures or summary data. It
  can be thought of as the "in-the-lab" portion of the experiment, where raw
  data is collected. For example, in the ACAS Xu experiments, this computes the
  classifications of the lines (but does *not* produce the actual plots).
- The "Analyze" phase takes the raw data produced by the run phase and creates
  figures and summary statistics ready for use in the paper.

The primary motivation for this separation of concerns is to be able to quickly
modify the analysis performed (eg. making small modifications to the generated
figures) without having to re-run the underlying experiments. This allows
faster iteration cycles and quicker investigation into experiment data.

To that end, after the run phase is executed, subsequent executions of the
experiment will allow you to (optionally) skip the run phase and only execute
the analysis.

There is also a ``main`` method which handles orchestration of the two other
phases.

When first reading an Experiment, we suggest starting with the ``run`` and
``analyze`` methods, which are the main entry points to the class.

### Data Collection and Artifact Storage
Furthermore, there is a standard interface provided for data collection.

At a high-level, each experiment produces and keeps track of a list of
"artifacts" (eg. Numpy arrays or MatPlotLib plots), which can be recorded using
the method ``Experiment.record_artifact`` and accessed during the analysis
phase using ``Experiment.read_artifact``. Each artifact has a key, type, and
associated value. The ``record_artifact`` and ``read_artifact`` methods
transparently handle serialization and deserialization when applicable.

There are also CSV file helpers ``Experiment.begin_csv``,
``Experiment.write_csv``, and ``Experiment.read_csv`` which treat the CSV file
like a dictionary of strings. CSV file keys can be stored as artifacts for
future retrieval.

After each phase, all artifacts are compressed into an tar-gz archive that can
be later read from using ``Experiment.open`` or any standard archive reader.
