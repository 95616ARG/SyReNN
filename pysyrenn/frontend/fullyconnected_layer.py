"""Methods for describing a Fully-Connected layer.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class FullyConnectedLayer(NetworkLayer):
    """Represents a fully-connected (arbitrary affine) layer in a network.
    """
    def __init__(self, weights, biases):
        """Constructs a new FullyConnectedLayer.
        """
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        if biases is not None:
            self.biases = torch.tensor(biases, dtype=torch.float32)

    def compute(self, inputs, jacobian=False):
        """Returns the output of the layer on @inputs.

        If @jacobian=True, it only computes the homogeneous portion (i.e., does
        not add biases). This can be used to compute the product of the
        Jacobian of the layer with its inputs.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        output = torch.mm(inputs, self.weights)
        if not jacobian:
            output += self.biases
        if is_np:
            return output.numpy()
        return output

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.fullyconnected_data.weights.extend(
            list(self.weights.numpy().flatten()))
        serialized.fullyconnected_data.biases.extend(
            list(self.biases.numpy().flatten()))
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "fullyconnected_data":
            weights = np.array(serialized.fullyconnected_data.weights)
            biases = np.array(serialized.fullyconnected_data.biases)
            weights = weights.reshape((-1, len(biases)))
            return cls(weights, biases)
        return None
