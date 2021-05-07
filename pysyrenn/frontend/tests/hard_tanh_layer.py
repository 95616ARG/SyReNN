"""Tests the methods in hard_tanh_layer.py.
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.hard_tanh_layer import HardTanhLayer

def test_compute():
    """Tests that the HardTanh layer correctly computes a HardTanh.
    """
    inputs = np.random.uniform(size=(101, 1025))
    true_hard_tanh = np.minimum(np.maximum(inputs, -1.0), 1.0)

    layer = HardTanhLayer()
    assert np.allclose(layer.compute(inputs), true_hard_tanh)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, torch_inputs)

def test_serialize():
    """Tests that the HardTanh layer correctly [de]serializes itself.
    """
    serialized = HardTanhLayer().serialize()
    assert serialized.WhichOneof("layer_data") == "hard_tanh_data"

    deserialized = HardTanhLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.relu_data.SetInParent()
    assert HardTanhLayer.deserialize(serialized) is None

main(__name__, __file__)
