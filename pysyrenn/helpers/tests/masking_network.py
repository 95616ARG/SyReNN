"""Tests the methods in masking_network.py
"""
import numpy as np
import torch
import pytest
from pysyrenn.frontend.tests.helpers import main
from pysyrenn.frontend import ReluLayer, FullyConnectedLayer, ArgMaxLayer
from pysyrenn.helpers.masking_network import MaskingNetwork

def test_compute():
    """Tests that it works for a simple example.
    """
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    value_layers = activation_layers[:2] + [
        FullyConnectedLayer(3.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    network = MaskingNetwork(activation_layers, value_layers)
    assert network.differ_index == 2
    output = network.compute([[-2.0, 1.0]])
    assert np.allclose(output, [[0.0, 6.0]])

def test_nodiffer():
    """Tests the it works if activation and value layers are identical.
    """
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    value_layers = activation_layers
    network = MaskingNetwork(activation_layers, value_layers)
    assert network.differ_index == 4
    output = network.compute([[-2.0, 1.0]])
    assert np.allclose(output, [[0.0, 4.0]])

def test_bad_layer():
    """Tests that non-ReLU/FullyConnected layers only after differ_index fail.
    """
    # It should work if it's before the differ_index.
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ArgMaxLayer(),
    ]
    value_layers = activation_layers
    network = MaskingNetwork(activation_layers, value_layers)
    assert network.differ_index == 4
    output = network.compute([[-2.0, 1.0]])
    assert np.allclose(output, [[1.0]])
    # But not after the differ_index.
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ArgMaxLayer(),
    ]
    value_layers = activation_layers[:2] + [
        FullyConnectedLayer(3.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    network = MaskingNetwork(activation_layers, value_layers)
    assert network.differ_index == 2
    try:
        output = network.compute([[-2.0, 1.0]])
        assert False
    except NotImplementedError:
        pass

def test_serialization():
    """Tests that it correctly (de)serializes.
    """
    activation_layers = [
        FullyConnectedLayer(np.eye(2), np.ones(shape=(2,))),
        ReluLayer(),
        FullyConnectedLayer(2.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    value_layers = activation_layers[:2] + [
        FullyConnectedLayer(3.0 * np.eye(2), np.zeros(shape=(2,))),
        ReluLayer(),
    ]
    network = MaskingNetwork(activation_layers, value_layers)
    serialized = network.serialize()
    assert all(serialized == layer.serialize()
               for serialized, layer in zip(serialized.activation_layers,
                                            activation_layers))
    assert all(serialized == layer.serialize()
               for serialized, layer in zip(serialized.value_layers,
                                            value_layers[2:]))
    assert serialized.differ_index == 2

    assert MaskingNetwork.deserialize(serialized).serialize() == serialized

main(__name__, __file__)
