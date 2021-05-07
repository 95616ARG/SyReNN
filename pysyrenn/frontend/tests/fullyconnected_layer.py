"""Tests the methods in fullyconnected_layer.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.fullyconnected_layer import FullyConnectedLayer

def test_compute():
    """Tests that the Fully-Connected layer correctly computes.
    """
    n_inputs = 1025
    n_outputs = 2046
    batch = 15
    inputs = np.random.uniform(size=(batch, n_inputs))
    weights = np.random.uniform(size=(n_inputs, n_outputs))
    biases = np.random.uniform(size=(n_outputs))

    true_outputs = np.matmul(inputs, weights) + biases

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    assert np.allclose(fullyconnected_layer.compute(inputs), true_outputs)

    torch_inputs = torch.FloatTensor(inputs)
    torch_outputs = fullyconnected_layer.compute(torch_inputs).numpy()
    assert np.allclose(torch_outputs, true_outputs)

def test_serialize():
    """Tests that the Fully-Connected layer correctly [de]serializes itself.
    """
    n_inputs = 129
    n_outputs = 291
    weights = np.random.uniform(size=(n_inputs, n_outputs))
    biases = np.random.uniform(size=(n_outputs))

    serialized = FullyConnectedLayer(weights, biases).serialize()
    assert serialized.WhichOneof("layer_data") == "fullyconnected_data"

    serialized_weights = np.array(serialized.fullyconnected_data.weights)
    assert np.allclose(serialized_weights.flatten(), weights.flatten())

    serialized_biases = np.array(serialized.fullyconnected_data.biases)
    assert np.allclose(serialized_biases.flatten(), biases.flatten())

    deserialized = FullyConnectedLayer.deserialize(serialized)
    assert deserialized.serialize() == serialized

    serialized.relu_data.SetInParent()
    deserialized = FullyConnectedLayer.deserialize(serialized)
    assert FullyConnectedLayer.deserialize(serialized) is None

main(__name__, __file__)
