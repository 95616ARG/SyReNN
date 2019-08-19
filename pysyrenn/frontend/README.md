# SyReNN Python Frontend
This directory contains a thin Python front-end to the gRPC server in
[../syrenn_server].

## Directory Contents
1. ``network.py`` defines the ``Network`` class, which represents an entire
   network (effectively a list of sequentially-applied layers). It has helper
   methods for importing from standard file types as well as calling the
   server.
2. ``layer.py`` and ``*_layer.py`` define layers that can be added to a network
   and serialized for sending to the SyReNN server. Forward-passes can be
   computed using PyTorch to allow gradient computation.
3. ``transformer_client.py`` handles communication with the gRPC server; it is
   not part of the public API (instead use the corresponding methods on the
   ``Network`` class).
4. ``tests/{file}.py`` contains PyTest code for the methods in ``{file}.py``.
   They can be run with ``make pytest_coverage`` in the parent directory (see
   [../README.md](../README.md)).

## Usage
Before attempting to use the Python front-end, please ensure that there is a
server running on the machine (see [../README.md](../README.md)).

The primary interface is the ``Network`` class. You can import a Network from
the ``.eran`` file format used by [ERAN](https://github.com/eth-sri/eran) (note
--- not all models currently supported), the ``.onnx`` format (again, not all
models are compatible), or by manually constructing one using the layers
described in ``*_layer.py``.

### Importing a Model
#### From ERAN Format
The ERAN format is the preferred way to import models that use only
convolutional, fully-connected, and PWL-nonlinearities.

```python
from pysyrenn import Network

# Importing from ERAN format.
network = Network.from_file("model.eran", file_type="eran")
# "eran" file type is implied by the suffix ".eran":
network = Network.from_file("model.eran")
```

#### From ONNX Format
Models can also be imported from ONNX format:

```python
# Importing from ONNX format.
network = Network.from_file("model.onnx", file_type="onnx")
# "onnx" file type is implied by the suffix ".onnx":
network = Network.from_file("model.onnx")
```

However, please note that the ONNX importer is currently poorly-tested and may
not support all layer types/features from ONNX. Please double-check that your
ONNX model was load correctly by investigating ``network.layers`` and using
``network.compute`` to compare the loaded model against your expectations. If
you find any issues with ONNX imports, please contact us.

#### Directly from Layers
Finally, you can directly construct a ``Network`` using the Python API. This is
the best option if your program has its own model representation. Please refer
to the ``*_layer.py`` files in this directory for more information about
creating ``Layer``s, particularly as to the data format used.

```python
from pysyrenn import FullyConnectedLayer, ReluLayer
weights = ... # Numpy array, shape: (in_dims, out_dims)
biases = ... # Numpy array, shape: (out_dims,)
fullyconnected = FullyConnectedLayer(weights, biases)
relu = ReluLayer()
network = Network([fullyconnected, relu])
```

## Using a ``Network``
Once you have a ``Network`` instance, the following functions can be used to
interface with the SyReNN server:

- ``network.exactlines``
- ``network.exactline``
- ``network.transform_planes``
- ``network.transform_plane``

Note that the singular functions are simply wrappers around their plural
counterparts (see below).

### Test Cases and Experiments
Please refer to [../../experiments](../../experiments) for usage of PySyReNN,
along with [tests](tests) for the front-end test cases.

### ExactLines Usage Examples
#### Constructing a Network
First, we create a network with just a ReLU layer, along with three lines in
2-dimensional space described by their starting & ending coordinates:

```python
network = Network([ReluLayer()]) # See above.

input_lines = [
    # Line passing through three quadrants.
    ([-0.5, 1], [1, -0.5]),
    # Line passing through two quadrants.
    ([-1, 3], [2, 0]),
    # Line only in the first quadrant.
    ([0.5, 0.5], [0.25, 0.75]),
]
```

#### ``compute_preimages=False, include_post=False``
When ``compute_preimages=False`` and ``include_post=False``, ExactLines will
return the _ratios of the partition endpoints_ for each line:

```python
exactlines = network.exactlines(input_lines,
                                compute_preimages=False,
                                include_post=False)
# Each line in input_lines is transformed and returned.
assert len(exactlines) == len(input_lines)

# The first line passes through 3 quadrants, so has 4 endpoints.
assert len(exactlines[0]) == 4
# They happen to be equally distributed for this line.
assert np.allclose(exactlines[0], [0.0, 1.0/3.0, 2.0/3.0, 1.0])

# The next passes through 2 quadrants for 3 endpoints.
assert len(exactlines[1]) == 3
assert np.allclose(exactlines[1], [0.0, 1.0/3.0, 1.0])

# The last is only in the first quadrant, so we have just the original two
# endpoints.
assert len(exactlines[2]) == 2
assert np.allclose(exactlines[2], [0.0, 1.0])
```

#### ``compute_preimages=True``
``compute_preimages=True`` returns the actual (preimage) points along each
line, instead of their ratios:

```python
exactlines = network.exactlines(input_lines,
                                compute_preimages=True,
                                include_post=False)
assert len(exactlines) == 3

assert np.allclose(exactlines[0], [exactlines[0][0],
                                   [0, 0.5],
                                   [0.5, 0],
                                   exactlines[0][1]])

... # Similarly for exactlines[1] and exactlines[2]
```

#### ``include_post=True``
``include_post=True`` transforms each line into a tuple (pre, post) where pre
is equivalent to what you would get from calling ``exactlines`` with
``include_post=False`` (i.e. the endpoints) and post is the output of the
network on each endpoint.

```python
exactlines = network.exactlines(input_lines,
                                compute_preimages=False,
                                include_post=True)
assert len(exactlines) == 3

assert isinstance(exactlines[0], tuple) and len(exactlines[0]) == 2
pre, post = exactlines[0]
assert np.allclose(pre, [0.0, 1.0/3.0, 2.0/3.0, 1.0])
assert np.allclose(post, [[0.0, 1.0], [0.0, 0.5], [0.5, 0.0], [1.0, 0.0]])

... # Similarly for exactlines[1] and exactlines[2]
```

#### ``exactline`` (Singular Method)
In all cases, the singular ``exactline`` method does what you would expect,
except it takes ``start`` and ``end`` parameters instead of a tuple:
```python
exactlines = network.exactlines(input_lines, ...)
start, end = input_lines[0]
exactline = network.exactline(start, end, ...)
assert np.allclose(exactline, exactlines[0])
```

### Transform-Planes
Transforming planes is similar to ExactLine except planes are represented as
Numpy arrays of vertices, not tuples of (start, end)-points.
