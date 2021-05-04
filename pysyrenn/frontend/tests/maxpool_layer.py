"""Tests the methods in maxpool_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.strided_window_data import StridedWindowData
from pysyrenn.frontend.maxpool_layer import MaxPoolLayer

def test_compute():
    """Tests that the MaxPool layer correctly computes a MaxPool.
    """
    batch = 101
    width = 32
    height = 32
    channels = 3
    inputs = np.random.uniform(size=(101, height * width * channels))

    true_outputs = inputs.reshape((batch, height, width, channels))
    true_outputs = true_outputs.reshape((batch, height, width // 2, 2, channels))
    true_outputs = np.max(true_outputs, axis=3)
    true_outputs = true_outputs.reshape((batch, height // 2, 2, -1, channels))
    true_outputs = np.max(true_outputs, axis=2).reshape((batch, -1))

    window_data = StridedWindowData((height, width, channels),
                                    (2, 2), (2, 2), (0, 0), channels)
    maxpool_layer = MaxPoolLayer(window_data)
    assert np.allclose(maxpool_layer.compute(inputs), true_outputs)
    output, indices = maxpool_layer.compute(inputs, return_indices=True)
    assert np.allclose(output, true_outputs)
    # TODO: Actually check true_indices itself.
    assert np.allclose(maxpool_layer.from_indices(inputs, indices), output)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = maxpool_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_outputs)
    torch_outputs, torch_indices = maxpool_layer.compute(torch_inputs,
                                                         return_indices=True)
    assert np.allclose(torch_outputs.numpy(), true_outputs)
    torch_outputs = maxpool_layer.from_indices(torch_inputs, indices)
    assert np.allclose(torch_outputs.numpy(), true_outputs)

def test_serialize():
    """Tests that the MaxPool layer correctly [de]serializes itself.
    """
    height, width, channels = np.random.choice([8, 16, 32, 64, 128], size=3)
    window_height, window_width = np.random.choice([2, 4, 8], size=2)
    window_data = StridedWindowData((height, width, channels),
                                    (window_height, window_width),
                                    (window_height, window_width),
                                    (0, 0), channels)

    serialized = MaxPoolLayer(window_data).serialize()
    assert serialized.WhichOneof("layer_data") == "maxpool_data"

    serialized_window_data = serialized.maxpool_data.window_data
    assert serialized_window_data == window_data.serialize()

    deserialized = MaxPoolLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.relu_data.SetInParent()
    assert MaxPoolLayer.deserialize(serialized) is None

main(__name__, __file__)
