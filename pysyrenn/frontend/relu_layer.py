"""Methods describing ReLU layers in networks.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class ReluLayer(NetworkLayer):
    """Represents a rectified-linear layer in a network.
    """
    def compute(self, inputs):
        """Computes ReLU(@inputs).
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.nn.functional.relu(inputs)
        if is_np:
            return outputs.numpy()
        return outputs

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.relu_data.SetInParent()
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "relu_data":
            return cls()
        return None
