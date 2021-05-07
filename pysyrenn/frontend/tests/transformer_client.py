"""Tests the methods in network.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend import transformer_client
from pysyrenn.frontend.network import Network
from pysyrenn.frontend.conv2d_layer import Conv2DLayer
from pysyrenn.frontend.fullyconnected_layer import FullyConnectedLayer
from pysyrenn.frontend.relu_layer import ReluLayer
from pysyrenn.frontend.hard_tanh_layer import HardTanhLayer
from pysyrenn.frontend.normalize_layer import NormalizeLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class ServerStubMock:
    def __init__(self, response_messages):
        self.received_messages = []
        self.response_messages = response_messages

    def Transform(self, request_iterator):
        self.received_messages.append(list(request_iterator))
        return self.response_messages.copy()

def test_transform_lines():
    open_stub_ = transformer_client.open_stub

    input_dims = np.random.randint(1, 32)
    output_dims = np.random.randint(1, 64)

    weights = np.random.uniform(size=(input_dims, output_dims))
    biases = np.random.uniform(size=(output_dims))

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    relu_layer = ReluLayer()

    network = Network([fullyconnected_layer, relu_layer])
    lines = list(np.random.uniform(size=(100, 2, input_dims)))

    response_messages = []
    for i, line in enumerate(lines):
        transformed_line = transformer_pb.SegmentedLine()
        for j in range(i + 2):
            endpoint = transformer_pb.SegmentEndpoint()
            endpoint.coordinates.extend(np.arange(i, i + 10))
            endpoint.preimage_ratio = j / ((i + 2) - 1)
            transformed_line.endpoints.append(endpoint)
        response = transformer_pb.TransformResponse()
        response.transformed_line.CopyFrom(transformed_line)
        response_messages.append(response)

    # With include_post = True.
    stub = ServerStubMock(response_messages)
    transformer_client.open_stub = lambda: stub
    transformed_lines = transformer_client.transform_lines(network, lines,
                                                           include_post=True)
    def verify_response(stub, transformed_lines, included_post):
        assert len(transformed_lines) == len(lines)
        for i, line in enumerate(lines):
            transformed_pre, transformed_post = transformed_lines[i]
            assert len(transformed_pre) == (i + 2)
            assert np.allclose(transformed_pre,
                               [j / ((i + 2) - 1) for j in range(i + 2)])
            if included_post:
                assert len(transformed_post) == (i + 2)
                assert np.allclose(transformed_post, np.arange(i, i + 10))
            else:
                assert transformed_post is None
        assert len(stub.received_messages) == 1
        received = stub.received_messages[0]
        assert len(received) == 102
        assert received[0].WhichOneof("request_data") == "layer"
        assert received[0].layer == fullyconnected_layer.serialize()
        assert received[1].layer == relu_layer.serialize()
        for i, request in enumerate(received[2:]):
            assert request.WhichOneof("request_data") == "line"
            assert len(request.line.endpoints) == 2
            assert request.line.endpoints[0].preimage_ratio == 0.0
            assert request.line.endpoints[1].preimage_ratio == 1.0
            assert np.allclose(np.array(request.line.endpoints[0].coordinates),
                               lines[i][0])
            assert np.allclose(np.array(request.line.endpoints[1].coordinates),
                               lines[i][1])
    verify_response(stub, transformed_lines, True)

    # With include_post = False.
    for response_message in response_messages:
        for endpoint in response_message.transformed_line.endpoints:
            while endpoint.coordinates:
                endpoint.coordinates.pop()
    stub = ServerStubMock(response_messages)
    transformer_client.open_stub = lambda: stub
    transformed_lines = transformer_client.transform_lines(network, lines,
                                                           include_post=False)
    verify_response(stub, transformed_lines, False)
    transformer_client.open_stub = open_stub_

def test_transform_planes():
    open_stub_ = transformer_client.open_stub

    input_dims = np.random.randint(1, 32)
    output_dims = np.random.randint(1, 64)

    weights = np.random.uniform(size=(input_dims, output_dims))
    biases = np.random.uniform(size=(output_dims))

    fullyconnected_layer = FullyConnectedLayer(weights, biases)
    relu_layer = ReluLayer()

    network = Network([fullyconnected_layer, relu_layer])
    planes = list(np.random.uniform(size=(100, 3, input_dims)))

    response_messages = []
    for i, plane in enumerate(planes):
        transformed_polytope = transformer_pb.UPolytope()
        transformed_polytope.space_dimensions = output_dims
        transformed_polytope.subspace_dimensions = 2
        for j in range(i + 2):
            polytope = transformer_pb.VPolytope()
            polytope.vertices.extend(np.matmul(plane, weights).flatten())
            polytope.combinations.extend(np.eye(3).flatten())
            polytope.num_vertices = 3
            transformed_polytope.polytopes.append(polytope)
        response = transformer_pb.TransformResponse()
        response.transformed_upolytope.CopyFrom(transformed_polytope)
        response_messages.append(response)

    # With include_post = True.
    stub = ServerStubMock(response_messages)
    transformer_client.open_stub = lambda: stub
    transformed = transformer_client.transform_planes(network, planes)

    assert len(transformed) == len(planes)
    for i, plane in enumerate(planes):
        upolytope = transformed[i]
        assert len(upolytope) == (i + 2)
        for vpolytope in upolytope:
            transformed_pre, transformed_post = vpolytope
            assert len(transformed_pre) == len(transformed_post) == 3
            assert np.allclose(transformed_pre, np.eye(3))
            assert np.allclose(transformed_post, np.matmul(plane, weights))
    assert len(stub.received_messages) == 1
    received = stub.received_messages[0]
    assert len(received) == 102
    assert received[0].WhichOneof("request_data") == "layer"
    assert received[0].layer == fullyconnected_layer.serialize()
    assert received[1].layer == relu_layer.serialize()
    for i, request in enumerate(received[2:]):
        assert request.WhichOneof("request_data") == "upolytope"
        assert request.upolytope.space_dimensions == input_dims
        assert request.upolytope.subspace_dimensions == 2

        assert len(request.upolytope.polytopes) == 1
        assert request.upolytope.polytopes[0].num_vertices == 3
        assert np.allclose(request.upolytope.polytopes[0].vertices,
                           planes[i].flatten())
        assert np.allclose(request.upolytope.polytopes[0].combinations,
                           np.eye(3).flatten())

main(__name__, __file__)
