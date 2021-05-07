"""Methods describing concatenation layers.
"""
import aenum
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class ConcatAlong(aenum.Enum):
    """Specifies how the concatenation should be performed.
    """
    # Concatenate the channels (assumes height/width all identical)
    CHANNELS = aenum.auto()
    # Concatenate the flattened activation vectors.
    FLAT = aenum.auto()

    def serialize(self):
        """Serializes the ConcatAlong type for the transformer server.
        """
        full_name = "CONCAT_ALONG_{}".format(self.name)
        return transformer_pb.ConcatLayerData.ConcatAlong.Value(full_name)

    @classmethod
    def deserialize(cls, serialized):
        if serialized == 1:
            return ConcatAlong.CHANNELS
        if serialized == 2:
            return ConcatAlong.FLAT
        raise NotImplementedError

# pylint: disable=abstract-method
class ConcatLayer(NetworkLayer):
    """Represents a concat layer in a network.
    """
    def __init__(self, input_layers, concat_along=ConcatAlong.CHANNELS):
        """Constructs a new ConcatLayer.

        input_layers should be a list of layers that are computed then
        concatenated into the final output of this layer. The layers in
        input_layers should *NOT* be included directly in the Network instance;
        they are represented and executed by the corresponding ConcatLayer.
        """
        self.input_layers = input_layers
        self.concat_along = concat_along

    def compute(self, inputs):
        """Computes the input layers and concatenation given an input vector.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        if self.concat_along == ConcatAlong.CHANNELS:
            assert all(hasattr(layer, "window_data")
                       for layer in self.input_layers)
            computed_inputs = [
                layer.compute(inputs).reshape(
                    (-1, layer.window_data.out_channels))
                for layer in self.input_layers]
        elif self.concat_along == ConcatAlong.FLAT:
            computed_inputs = [layer.compute(inputs)
                               for layer in self.input_layers]
        else:
            raise NotImplementedError
        concatenated = torch.cat(computed_inputs, dim=1)
        concatenated = concatenated.reshape((inputs.shape[0], -1))
        if is_np:
            return concatenated.numpy()
        return concatenated

    def serialize(self):
        """Serializes the layer for the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.concat_data.layers.extend([
            input_layer.serialize() for input_layer in self.input_layers
        ])
        serialized.concat_data.concat_along = self.concat_along.serialize()
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer.
        """
        if serialized.WhichOneof("layer_data") == "concat_data":
            from pysyrenn.frontend.network import Network
            layers = Network.deserialize_layers(serialized.concat_data.layers)
            concat_along = ConcatAlong.deserialize(
                serialized.concat_data.concat_along)
            return cls(layers, concat_along)
        return None
