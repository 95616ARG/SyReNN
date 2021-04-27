"""Tests the methods in normalize_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.normalize_layer import NormalizeLayer

def test_compute():
    """Tests that the Normalize layer correctly computes.
    """
    dims = 1025
    batch = 15

    inputs = np.random.uniform(size=(batch, dims)).astype(np.float32)

    means = np.random.uniform(size=(dims)).astype(np.float32)
    stds = np.random.uniform(size=(dims)).astype(np.float32)

    true_outputs = (inputs - means) / stds

    normalize_layer = NormalizeLayer(means, stds)
    assert np.allclose(normalize_layer.compute(inputs), true_outputs)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = normalize_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_outputs)

def test_serialize():
    """Tests that the Normalize layer correctly [de]serializes itself.
    """
    n_dims = 129
    means = np.random.uniform(size=(n_dims))
    stds = np.random.uniform(size=(n_dims))

    serialized = NormalizeLayer(means, stds).serialize()
    assert serialized.WhichOneof("layer_data") == "normalize_data"

    serialized_means = np.array(serialized.normalize_data.means)
    assert np.allclose(serialized_means.flatten(), means.flatten())

    serialized_stds = np.array(serialized.normalize_data.standard_deviations)
    assert np.allclose(serialized_stds.flatten(), stds.flatten())

    deserialized = NormalizeLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.relu_data.SetInParent()
    assert NormalizeLayer.deserialize(serialized) is None

main(__name__, __file__)
