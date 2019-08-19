"""Methods describing a normalization layer.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class NormalizeLayer(NetworkLayer):
    """Represents a normalization layer in a network.
    """
    def __init__(self, means, standard_deviations):
        """Constructs a new NormalizeLayer.
        """
        self.means = torch.tensor(means, dtype=torch.float32)
        self.standard_deviations = torch.tensor(standard_deviations, dtype=torch.float32)

    def compute(self, inputs):
        """Returns the normalized form of @inputs.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        # Here we assume channels-last ordering, as in ERAN
        n_inputs = inputs.shape[0]
        inputs = inputs.reshape((-1, len(self.means)))
        outputs = (inputs - self.means) / self.standard_deviations
        outputs = outputs.reshape((n_inputs, -1))
        if is_np:
            return outputs.numpy()
        return outputs

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.normalize_data.means.extend(
            list(self.means.numpy().flatten()))
        serialized.normalize_data.standard_deviations.extend(
            list(self.standard_deviations.numpy().flatten()))
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "normalize_data":
            means = np.array(serialized.normalize_data.means)
            stds = np.array(serialized.normalize_data.standard_deviations)
            return cls(means, stds)
        return None
