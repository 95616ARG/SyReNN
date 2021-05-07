"""Tests the methods in relu_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.relu_layer import ReluLayer

def test_compute():
    """Tests that the ReLU layer correctly computes a ReLU.
    """
    inputs = np.random.uniform(size=(101, 1025))
    true_relu = np.maximum(inputs, 0.0)

    relu_layer = ReluLayer()
    assert np.allclose(relu_layer.compute(inputs), true_relu)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = relu_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_relu)

def test_serialize():
    """Tests that the ReLU layer correctly [de]serializes itself.
    """
    serialized = ReluLayer().serialize()
    assert serialized.WhichOneof("layer_data") == "relu_data"

    deserialized = ReluLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.normalize_data.SetInParent()
    assert ReluLayer.deserialize(serialized) is None

main(__name__, __file__)
