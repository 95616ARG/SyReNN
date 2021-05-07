"""Tests the methods in argmax_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.argmax_layer import ArgMaxLayer

def test_compute():
    """Tests that the ArgMax layer correctly computes a ArgMax.
    """
    inputs = np.random.uniform(size=(101, 1025))
    true_argmax = np.argmax(inputs, axis=1)

    argmax_layer = ArgMaxLayer()
    assert np.allclose(argmax_layer.compute(inputs), true_argmax)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = argmax_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_argmax)

def test_serialize():
    """Tests that the layer correctly serializes/deserializes itself.
    """
    serialized = ArgMaxLayer().serialize()
    assert serialized.WhichOneof("layer_data") == "argmax_data"

    deserialized = ArgMaxLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.relu_data.SetInParent()
    deserialized = ArgMaxLayer.deserialize(serialized)
    assert deserialized is None

main(__name__, __file__)
