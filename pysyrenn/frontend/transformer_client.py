"""Helper methods for interacting with the gRPC LineTransformer server.

Client code should use the helper methods in the Network class, which call
these methods.
"""
import os
import numpy as np
import sys
sys.path = [path for path in sys.path if "/com_github_grpc_grpc/" not in path]
import grpc
import syrenn_proto.syrenn_pb2_grpc as grpc_pb
import syrenn_proto.syrenn_pb2 as transformer_pb

SERVER = "localhost:50051"
if "SYRENN_SERVER" in os.environ:
    SERVER = os.environ["SYRENN_SERVER"]

def open_channel():
    """Opens a channel to the SyReNN transformer server.
    """
    max_message = 32 * 4194304
    return grpc.insecure_channel(SERVER, options=[
        ('grpc.max_send_message_length', max_message),
        ('grpc.max_receive_message_length', max_message)])

def open_stub():
    """Returns a gRPC stub for the SyReNN transformer server.
    """
    return grpc_pb.SyReNNTransformerStub(open_channel())

def transform_lines(network, lines, include_post):
    """Returns the post-set of each input line in lines.

    Arguments
    =========
    - network should be a Network
    - lines should be a list of tuples (from_point, to_point), each one
      defining the line formed by interpolating betwen from_point and to_point.
    - include_post controls whether postimages are returned. It is slightly
      faster to use include_post=False if you only need the partitioning.

    If include_post=False, returns a list of list of endpoint-ratios (one list
    of endpoint-ratios for each line in @lines).
    Returns a list of tuples, where each line in @lines has a tuple
    (endpoint-ratios, post-endpoints) (with post-endpoints=None if
    include_post=False).
    """
    request_messages = []

    # First, we serialize the network.
    for layer in network.layers:
        request = transformer_pb.TransformRequest()
        request.layer.CopyFrom(layer.serialize())
        request_messages.append(request)

    # Then, we serialize the lines.
    for from_point, to_point in lines:
        request = transformer_pb.TransformRequest()

        from_endpoint = request.line.endpoints.add()
        from_endpoint.coordinates.extend(list(from_point))
        from_endpoint.preimage_ratio = 0.0

        to_endpoint = request.line.endpoints.add()
        to_endpoint.coordinates.extend(list(to_point))
        to_endpoint.preimage_ratio = 1.0

        request.include_post = include_post

        request_messages.append(request)

    # Send the request
    stub = open_stub()
    response_messages = stub.Transform(iter(request_messages))

    # Finally, we deserialize the response
    linewise_results = []
    for response_message in response_messages:
        line = response_message.transformed_line
        line_distances = np.array([endpoint.preimage_ratio
                                   for endpoint in line.endpoints])
        if include_post:
            post_vertices = np.array([np.array(endpoint.coordinates)
                                      for endpoint in line.endpoints])
            linewise_results.append((line_distances, post_vertices))
        else:
            linewise_results.append((line_distances, None))
    return linewise_results

def transform_planes(network, planes):
    """Returns the post-set of each input plane in planes.

    Arguments
    =========
    - network should be a Network
    - planes should be a list of Numpy arrays, each one a counter-clockwise
      list of vertices defining the plane.

    Returns a list, containing for each input plane a list of VPolytopes, with
    each VPolytope being represented by a tuple:
    (pre-combinations, post-vertices).
    """
    request_messages = []

    # First, we serialize network
    for layer in network.layers:
        request = transformer_pb.TransformRequest()
        request.layer.CopyFrom(layer.serialize())
        request_messages.append(request)

    # Then, we serialize the planes
    for vertices in planes:
        request = transformer_pb.TransformRequest()

        request.upolytope.space_dimensions = vertices.shape[1]
        request.upolytope.subspace_dimensions = 2
        vpolytope = request.upolytope.polytopes.add()
        vpolytope.num_vertices = vertices.shape[0]
        vpolytope.vertices.extend(list(vertices.reshape([-1])))
        vpolytope.combinations.extend(np.eye(vertices.shape[0]).flatten())

        request_messages.append(request)

    stub = open_stub()
    response_messages = stub.Transform(iter(request_messages))

    # Finally, we deserialize the response
    deserialized_planes = []
    for response_message in response_messages:
        upolytope = response_message.transformed_upolytope
        deserialized_plane = []
        for vpolytope in upolytope.polytopes:
            vertices = np.array(vpolytope.vertices)
            vertices = vertices.reshape((vpolytope.num_vertices, -1))
            combinations = np.array(vpolytope.combinations)
            combinations = combinations.reshape((vpolytope.num_vertices, -1))
            deserialized_plane.append((combinations, vertices))
        deserialized_planes.append(deserialized_plane)
    return deserialized_planes
