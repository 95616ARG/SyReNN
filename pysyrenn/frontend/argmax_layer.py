"""Methods describing an ArgMax layer.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class ArgMaxLayer(NetworkLayer):
    """Represents an ArgMax layer in a network.
    """
    def compute(self, inputs):
        """Returns ArgMax(inputs).
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        _, indices = torch.max(inputs, 1)
        if is_np:
            return indices.numpy()
        return indices

    def serialize(self):
        """Serializes the layer for the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.argmax_data.SetInParent()
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "argmax_data":
            return cls()
        return None
