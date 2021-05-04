"""Tests the methods in concat_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.strided_window_data import StridedWindowData
from pysyrenn.frontend.fullyconnected_layer import FullyConnectedLayer
from pysyrenn.frontend.normalize_layer import NormalizeLayer
from pysyrenn.frontend.conv2d_layer import Conv2DLayer
from pysyrenn.frontend.averagepool_layer import AveragePoolLayer
from pysyrenn.frontend.concat_layer import ConcatLayer, ConcatAlong
import syrenn_proto.syrenn_pb2 as transformer_pb

def test_compute_flat():
    """Tests that the Concat layer correctly computes.

    Uses concat_along = FLAT
    """
    batch = 15
    n_inputs = 1025
    fullyconnected_outputs = 2046

    inputs = np.random.uniform(size=(batch, n_inputs)).astype(np.float32)

    weights = np.random.uniform(size=(n_inputs, fullyconnected_outputs))
    weights = weights.astype(np.float32)
    biases = np.random.uniform(size=(fullyconnected_outputs))
    biases = biases.astype(np.float32)
    true_fullyconnected_outputs = np.matmul(inputs, weights) + biases

    means = np.random.uniform(size=(n_inputs)).astype(np.float32)
    stds = np.random.uniform(size=(n_inputs)).astype(np.float32)
    true_normalize_outputs = (inputs - means) / stds

    true_outputs = np.concatenate([true_fullyconnected_outputs,
                                   true_normalize_outputs],
                                  axis=1)

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    normalize_layer = NormalizeLayer(means, stds)

    concat_layer = ConcatLayer([fullyconnected_layer, normalize_layer],
                               ConcatAlong.FLAT)
    assert np.allclose(concat_layer.compute(inputs), true_outputs)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = concat_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_outputs)

def test_compute_channels():
    """Tests that the Concat layer correctly computes.

    Uses concat_along = CHANNELS
    """
    batch = 15
    height, width, channels = (32, 32, 3)
    out_channels = 6

    inputs = np.random.uniform(size=(batch, height*width*channels))
    inputs = inputs.astype(np.float32)

    filters = np.random.uniform(size=(2, 2, channels, out_channels))
    filters = filters.astype(np.float32)
    biases = np.random.uniform(size=(out_channels)).astype(np.float32)

    conv_window_data = StridedWindowData((height, width, channels),
                                         (2, 2), (2, 2), (0, 0), out_channels)
    conv2d_layer = Conv2DLayer(conv_window_data, filters, biases)
    conv2d_outputs = conv2d_layer.compute(inputs)

    pool_window_data = StridedWindowData((height, width, channels),
                                         (2, 2), (2, 2), (0, 0), channels)
    averagepool_layer = AveragePoolLayer(pool_window_data)
    pool_outputs = averagepool_layer.compute(inputs)

    true_outputs = np.concatenate([conv2d_outputs.reshape((-1, out_channels)),
                                   pool_outputs.reshape((-1, channels))],
                                  axis=1).reshape((batch, -1))

    concat_layer = ConcatLayer([conv2d_layer, averagepool_layer],
                               ConcatAlong.CHANNELS)
    assert np.allclose(concat_layer.compute(inputs), true_outputs)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = concat_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_outputs)

def test_compute_invalid():
    """Tests that the Concat layer fails on an invalid concat_along.
    """
    batch = 15
    n_inputs = 1025
    fullyconnected_outputs = 2046

    inputs = np.random.uniform(size=(batch, n_inputs)).astype(np.float32)

    weights = np.random.uniform(size=(n_inputs, fullyconnected_outputs))
    weights = weights.astype(np.float32)
    biases = np.random.uniform(size=(fullyconnected_outputs))
    biases = biases.astype(np.float32)
    fullyconnected_layer = FullyConnectedLayer(weights, biases)

    means = np.random.uniform(size=(n_inputs)).astype(np.float32)
    stds = np.random.uniform(size=(n_inputs)).astype(np.float32)
    normalize_layer = NormalizeLayer(means, stds)

    concat_layer = ConcatLayer([fullyconnected_layer, normalize_layer], None)
    try:
        concat_layer.compute(inputs)
        assert False
    except NotImplementedError:
        assert True

def test_serialize():
    """Tests that the Concat layer correctly serializes itself.
    """
    n_inputs = 125
    fullyconnected_outputs = 246

    weights = np.random.uniform(size=(n_inputs, fullyconnected_outputs))
    biases = np.random.uniform(size=(fullyconnected_outputs))
    fullyconnected_layer = FullyConnectedLayer(weights, biases)

    means = np.random.uniform(size=(n_inputs)).astype(np.float32)
    stds = np.random.uniform(size=(n_inputs)).astype(np.float32)
    normalize_layer = NormalizeLayer(means, stds)

    concat_layer = ConcatLayer([fullyconnected_layer, normalize_layer],
                               ConcatAlong.FLAT)
    serialized = concat_layer.serialize()
    assert serialized.WhichOneof("layer_data") == "concat_data"

    assert (serialized.concat_data.concat_along ==
            transformer_pb.ConcatLayerData.ConcatAlong.Value(
                "CONCAT_ALONG_FLAT"))
    assert len(serialized.concat_data.layers) == 2
    assert serialized.concat_data.layers[0] == fullyconnected_layer.serialize()
    assert serialized.concat_data.layers[1] == normalize_layer.serialize()

    # TODO: This does not check that deserialized.input_layers was done
    # correctly, but that should be the case as long as their deserialize
    # methods work (tested in their respective files).
    deserialized = ConcatLayer.deserialize(serialized)
    assert deserialized.concat_along == ConcatAlong.FLAT

    serialized.relu_data.SetInParent()
    deserialized = ConcatLayer.deserialize(serialized)
    assert deserialized is None

    concat_layer.concat_along = ConcatAlong.CHANNELS
    deserialized = ConcatLayer.deserialize(concat_layer.serialize())
    assert deserialized.concat_along == ConcatAlong.CHANNELS

    try:
        ConcatAlong.deserialize(5)
        assert False, "Should have errored on unrecognized serialization."
    except NotImplementedError:
        pass

main(__name__, __file__)
