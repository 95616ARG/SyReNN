"""Methods describing a Hard-Tanh layer.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class HardTanhLayer(NetworkLayer):
    """Represents a Hard-Tanh layer in a network.
    """
    def compute(self, inputs):
        """Computes the Hard-Tanh of @inputs.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = torch.nn.functional.hardtanh(inputs)
        if is_np:
            return outputs.numpy()
        return outputs

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.hard_tanh_data.SetInParent()
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "hard_tanh_data":
            return cls()
        return None
