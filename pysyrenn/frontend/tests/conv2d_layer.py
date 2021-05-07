"""Tests the methods in conv2d_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.strided_window_data import StridedWindowData
from pysyrenn.frontend.conv2d_layer import Conv2DLayer

def test_compute():
    """Tests that the Conv2D layer correctly computes a Conv2D.
    """
    batch = 101
    width = 32
    height = 32
    channels = 3

    stride = (2, 2)
    pad = (0, 0)
    filter_height = 4
    filter_width = 4
    out_channels = 5

    inputs = np.random.uniform(size=(101, height * width * channels))

    # TODO(masotoud): use actual numbers for the filters and actually compute
    # true_outputs.
    filters = np.zeros(shape=(filter_height, filter_width, channels, out_channels))
    biases = np.ones(shape=(out_channels))
    # out height/width = (32 - 2) / 2 = 15
    true_outputs = np.ones(shape=(batch, 15 * 15 * out_channels))

    window_data = StridedWindowData((height, width, channels),
                                    (filter_height, filter_width),
                                    stride, pad, out_channels)
    conv2d_layer = Conv2DLayer(window_data, filters, biases)
    assert np.allclose(conv2d_layer.compute(inputs), true_outputs)
    assert np.allclose(conv2d_layer.compute(inputs, jacobian=True),
                       np.zeros_like(true_outputs))

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = conv2d_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_outputs)

def test_serialize():
    """Tests Conv2D.{serialize, deserialize}.py.
    """
    height, width, channels, out_channels = np.random.choice(
            [8, 16, 32, 64, 128], size=4)
    window_height, window_width = np.random.choice([2, 4, 8], size=2)
    pad = (0, 0)
    window_data = StridedWindowData((height, width, channels),
                                    (window_height, window_width),
                                    (window_height, window_width),
                                    pad, out_channels)

    filters = np.random.uniform(size=(window_height, window_width,
                                      channels, out_channels))
    biases = np.random.uniform(size=(out_channels))
    serialized = Conv2DLayer(window_data, filters, biases).serialize()
    assert serialized.WhichOneof("layer_data") == "conv2d_data"

    serialized_window_data = serialized.conv2d_data.window_data
    assert serialized_window_data.in_height == height
    assert serialized_window_data.in_width == width
    assert serialized_window_data.in_channels == channels
    assert serialized_window_data.window_height == window_height
    assert serialized_window_data.window_width == window_width
    assert serialized_window_data.stride_height == window_height
    assert serialized_window_data.stride_width == window_width
    assert serialized_window_data.pad_height == 0
    assert serialized_window_data.pad_width == 0
    assert serialized_window_data.out_channels == out_channels

    serialized_filters = np.array(serialized.conv2d_data.filters)
    assert np.allclose(serialized_filters.flatten(), filters.flatten())

    serialized_biases = np.array(serialized.conv2d_data.biases)
    assert np.allclose(serialized_biases.flatten(), biases.flatten())

    deserialized = Conv2DLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.relu_data.SetInParent()
    assert Conv2DLayer.deserialize(serialized) is None

main(__name__, __file__)
