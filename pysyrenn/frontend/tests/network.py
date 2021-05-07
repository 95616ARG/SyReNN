"""Tests the methods in network.py
"""
import numpy as np
import torch
import pytest
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.network import Network
from pysyrenn.frontend.conv2d_layer import Conv2DLayer
from pysyrenn.frontend.fullyconnected_layer import FullyConnectedLayer
from pysyrenn.frontend.relu_layer import ReluLayer
from pysyrenn.frontend.hard_tanh_layer import HardTanhLayer
from pysyrenn.frontend.normalize_layer import NormalizeLayer
from pysyrenn.frontend.argmax_layer import ArgMaxLayer

def test_compute_and_gradients():
    """Tests the Network's compute and compute_gradients methods.
    """
    batch = np.random.randint(1, 128)
    input_dims = np.random.randint(1, 256)
    output_dims = np.random.randint(1, 512)

    inputs = np.random.uniform(size=(batch, input_dims))

    weights = np.random.uniform(size=(input_dims, output_dims))
    biases = np.random.uniform(size=(output_dims))

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    relu_layer = ReluLayer()

    fullyconnected_outputs = fullyconnected_layer.compute(inputs)
    relu_outputs = relu_layer.compute(fullyconnected_outputs)

    network = Network([fullyconnected_layer, relu_layer])
    network_outputs = network.compute(inputs)
    assert np.allclose(network_outputs, relu_outputs)
    assert np.allclose(network_outputs, network.compute(list(inputs)))
    assert np.allclose(network_outputs[0], network.compute(list(inputs)[0]))

    for label in range(output_dims):
        gradients = network.compute_gradients(inputs, label)
        for i in range(batch):
            if fullyconnected_outputs[i, label] <= 0.0:
                assert np.allclose(gradients[i], 0.0)
            else:
                assert np.allclose(gradients[i], weights[:, label])

def test_serialize():
    """Tests the Network's serialize and deserialize methods.
    """
    input_dims = np.random.randint(1, 32)
    output_dims = np.random.randint(1, 64)

    weights = np.random.uniform(size=(input_dims, output_dims))
    biases = np.random.uniform(size=(output_dims))

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    relu_layer = ReluLayer()

    network = Network([fullyconnected_layer, relu_layer])
    serialized = network.serialize()
    assert len(serialized.layers) == 2
    assert serialized.layers[0] == fullyconnected_layer.serialize()
    assert serialized.layers[1] == relu_layer.serialize()

    deserialized = Network.deserialize(serialized)
    assert deserialized.serialize() == serialized

def test_exactlines():
    import pysyrenn.frontend.transformer_client
    transform_lines_ = pysyrenn.frontend.transformer_client.transform_lines

    input_dims = np.random.randint(1, 32)
    output_dims = np.random.randint(1, 64)

    weights = np.random.uniform(size=(input_dims, output_dims))
    biases = np.random.uniform(size=(output_dims))

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    relu_layer = ReluLayer()

    network = Network([fullyconnected_layer, relu_layer])
    lines = list(np.random.uniform(size=(100, 2, input_dims)))

    def transform_lines_mock(query_network, query_lines,
                             query_include_post=False):
        assert query_network.serialize() == network.serialize()
        if len(query_lines) == 1:
            assert np.allclose(query_lines, lines[:1])
        else:
            assert np.allclose(query_lines, lines)
        output_lines = []
        for i, line in enumerate(query_lines):
            output_lines.append((np.array([0.0, 1.0 / float(i + 1), 1.0]),
                                 np.array([float(2.0 * i)])))
        return output_lines
    pysyrenn.frontend.transformer_client.transform_lines = transform_lines_mock

    ratios = network.exactlines(lines, compute_preimages=False,
                                include_post=False)
    assert np.allclose(ratios, np.array([[0.0, 1.0 / float(i + 1), 1.0]
                                         for i in range(100)]))
    ratio = network.exactline(*lines[0], compute_preimages=False,
                              include_post=False)
    assert np.allclose(ratio, ratios[0])


    def interpolate(line_i, ratio):
        start, end = lines[line_i]
        return start + (ratio * (end - start))
    preimages = network.exactlines(lines, compute_preimages=True,
                                   include_post=False)
    assert np.allclose(preimages, np.array([[interpolate(i, 0.0),
                                             interpolate(i, 1.0 / float(i + 1)),
                                             interpolate(i, 1.0)]
                                            for i in range(100)]))
    preimage = network.exactline(*lines[0], compute_preimages=True,
                                 include_post=False)
    assert np.allclose(preimage, preimages[0])

    transformed = network.exactlines(lines, compute_preimages=True,
                                     include_post=True)
    pre, post = zip(*transformed)
    assert np.allclose(pre, np.array([[interpolate(i, 0.0),
                                       interpolate(i, 1.0 / float(i + 1)),
                                       interpolate(i, 1.0)]
                                      for i in range(100)]))
    assert np.allclose(post, np.array([[float(2.0 * i)] for i in range(100)]))
    transformed_single = network.exactline(*lines[0], compute_preimages=True,
                                           include_post=True)
    assert np.allclose(transformed_single[0], transformed[0][0])
    assert np.allclose(transformed_single[1], transformed[0][1])

def test_fcn_from_eran():
    """Tests loading a fully-connected Network from ERAN format.
    """
    path = "fcn_test.eran"
    with open(path, "w") as netfile:
        netfile.write("ReLU\n[[-1, 2, -3], [-4, 5, -6]]\n[7, 8]\n")
        netfile.write("Normalize mean=[1, 2] std=[3, 4]\n")
        netfile.write("HardTanh\n[[-8, 7, -6], [-5, 4, -3]]\n[2, 1]\n")
    network = Network.from_file(path)
    assert len(network.layers) == 5
    assert isinstance(network.layers[0], FullyConnectedLayer)
    assert np.allclose(network.layers[0].weights,
                       np.array([[-1, -4], [2, 5], [-3, -6]]))
    assert np.allclose(network.layers[0].biases, np.array([[7, 8]]))
    assert isinstance(network.layers[1], ReluLayer)
    assert isinstance(network.layers[2], NormalizeLayer)
    assert np.allclose(network.layers[2].means, np.array([[1, 2]]))
    assert np.allclose(network.layers[2].standard_deviations,
                       np.array([[3, 4]]))
    assert isinstance(network.layers[3], FullyConnectedLayer)
    assert np.allclose(network.layers[3].weights,
                       np.array([[-8, -5], [7, 4], [-6, -3]]))
    assert np.allclose(network.layers[3].biases, np.array([[2, 1]]))
    assert isinstance(network.layers[4], HardTanhLayer)

def test_conv_from_eran():
    """Tests loading a convolutional Network from ERAN format.
    """
    path = "conv_test.eran"
    with open(path, "w") as netfile:
        netfile.write("Conv2D\nReLU, filters=2, kernel_size=[2, 2], ")
        netfile.write("input_shape=[16, 16, 2], stride=[10, 10], padding=2\n")
        netfile.write("[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],")
        netfile.write(" [[[8, 7], [6, 5]], [[4, 3], [2, 1]]]]\n")
        netfile.write("[-1, -2]\n")
        netfile.write("Affine\n[[1, 2, 3, 4, 5, 6, 7, 8], ")
        netfile.write("[5, 6, 7, 8, 9, 10, 11, 12]]\n[-1, -2]\n")
        netfile.write("Conv2D\nHardTanh, filters=1, kernel_size=[1, 1], ")
        netfile.write("input_shape=[1, 1, 2], stride=[1, 1], padding=0\n")
        netfile.write("[[[[1], [2]]]]\n")
        netfile.write("[-10]\n")
    network = Network.from_file(path)
    assert len(network.layers) == 5
    assert isinstance(network.layers[0], Conv2DLayer)
    assert np.allclose(network.layers[0].filter_weights,
                       np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                 [[[8, 7], [6, 5]], [[4, 3], [2, 1]]]]))
    assert np.allclose(network.layers[0].biases, np.array([-1, -2]))
    assert network.layers[0].window_data.input_shape == (16, 16, 2)
    assert network.layers[0].window_data.window_shape == (2, 2)
    assert network.layers[0].window_data.strides == (10, 10)
    assert network.layers[0].window_data.padding == (2, 2)
    assert network.layers[0].window_data.out_channels == 2
    assert isinstance(network.layers[1], ReluLayer)
    assert isinstance(network.layers[2], FullyConnectedLayer)
    assert np.allclose(network.layers[2].weights,
                       np.array([[1, 5, 2, 6, 3, 7, 4, 8],
                                 [5, 9, 6, 10, 7, 11, 8, 12]]).T)
    assert np.allclose(network.layers[2].biases, np.array([[-1, -2]]))
    assert isinstance(network.layers[3], Conv2DLayer)
    assert np.allclose(network.layers[3].filter_weights,
                       np.array([[[[1], [2]]]]))
    assert np.allclose(network.layers[3].biases, np.array([-10]))
    assert network.layers[3].window_data.input_shape == (1, 1, 2)
    assert network.layers[3].window_data.window_shape == (1, 1)
    assert network.layers[3].window_data.strides == (1, 1)
    assert network.layers[3].window_data.padding == (0, 0)
    assert network.layers[3].window_data.out_channels == 1
    assert isinstance(network.layers[4], HardTanhLayer)

def test_eran_unimplemented():
    """Tests loading a convolutional Network from ERAN format.
    """
    path = "eran_unimplemented.eran"
    with open(path, "w") as netfile:
        netfile.write("Sin\n")
        netfile.write("[[1, 2], [3, 4]]")
    try:
        network = Network.from_file(path)
        assert False
    except NotImplementedError:
        assert True
    with open(path, "w") as netfile:
        netfile.write("Conv2D\nSin, filters=1, kernel_size=[1, 1], ")
        netfile.write("input_shape=[1, 1, 2], stride=[1, 1], padding=0\n")
        netfile.write("[[[[1], [2]]]]\n")
        netfile.write("[-10]\n")
    try:
        network = Network.from_file(path)
        assert False
    except NotImplementedError:
        assert True

def test_squeezenet_from_onnx():
    """Tests loading a SqueezeNet Network from ONNX format.
    """
    network = Network.from_file("external/onnx_squeezenet/squeezenet1.1.onnx")
    assert len(network.layers) == 40

main(__name__, __file__)
