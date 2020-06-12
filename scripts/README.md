# Miscellaneous Scripts
This directory contains miscellaneous scripts related to SyReNN which we do not
yet want to make a part of the main packages and/or build system.

There is currently only one script, described below.

### Keras To SyReNN
`keras_to_syrenn.py` contains a `keras_to_syrenn(...)` method which shows how
to convert a sequential Keras Model into an equivalent SyReNN Network.

There is an example script converting a simple model in
`keras_to_syrenn_example.py`.

Please note that this script makes a lot of assumptions about the model,
including that it is sequential and uses HWIO layout for images. You should
always validate that the resultant SyReNN model has the same (within an
epsilon) output as the original Keras model.

We do not plan to move this code into the main SyReNN code in the near future,
as it does not currently support all networks that the current ONNX importer
does (e.g. with Concatenate layers), and it adds a heavy Tensorflow dependency
which we otherwise do not need.
