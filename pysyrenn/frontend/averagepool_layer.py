"""Methods describing AveragePool layers.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
from pysyrenn.frontend.strided_window_data import StridedWindowData
import syrenn_proto.syrenn_pb2 as transformer_pb

# pylint: disable=abstract-method
class AveragePoolLayer(NetworkLayer):
    """Represents an AveragePool layer in a network.

    Currently only supports pooling without padding and only H/W pooling,
    and only supports forward computation. This is mostly a stub class --- the
    only times we use it right now are for networks that we just pass to the
    line transformer anyways.
    """
    def __init__(self, window_data):
        """Constructs a new AveragePoolLayer.
        """
        self.window_data = window_data

    def compute(self, inputs):
        """Computes the AveragePool of inputs.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        # ERAN gives us NHWC
        inputs = self.window_data.unflatten_inputs(inputs)
        output_batches = inputs.shape[0]
        # PyTorch takes NCHW
        inputs = inputs.permute((0, 3, 1, 2))
        # Compute the avgpool
        output = torch.nn.functional.avg_pool2d(inputs,
                                                self.window_data.window_shape,
                                                self.window_data.strides,
                                                self.window_data.padding)
        # Pytorch gives us NCHW, ERAN uses NHWC
        output = output.permute((0, 2, 3, 1))
        output = output.reshape((output_batches, -1))
        if is_np:
            return output.numpy()
        return output

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.averagepool_data.window_data.CopyFrom(
            self.window_data.serialize())
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "averagepool_data":
            window_data = StridedWindowData.deserialize(
                serialized.averagepool_data.window_data)
            return cls(window_data)
        return None
