"""Methods for describing a MaxPool layer.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
from pysyrenn.frontend.strided_window_data import StridedWindowData
import syrenn_proto.syrenn_pb2 as transformer_pb

class MaxPoolLayer(NetworkLayer):
    """Represents a MaxPool layer in a network.
    """
    def __init__(self, window_data):
        """Constructs a new MaxPoolLayer.
        """
        assert len(window_data.window_shape) == 2
        self.window_data = window_data

    def compute(self, inputs, return_indices=False):
        """Computes the MaxPool given an input vector.

        If @return_indices=True, it returns a tuple (output, indices) where
        @indices specifies which input index each output came from. It can
        later be used in self.from_indices.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        # ERAN gives us NHWC
        inputs = self.window_data.unflatten_inputs(inputs)
        output_batches = inputs.shape[0]
        # PyTorch takes NCHW
        inputs = inputs.permute((0, 3, 1, 2))
        # Compute the maxpool
        output = torch.nn.functional.max_pool2d(inputs,
                                                self.window_data.window_shape,
                                                self.window_data.strides,
                                                self.window_data.padding,
                                                return_indices=return_indices)
        if return_indices:
            output, indices = output
        # Pytorch gives us NCHW, ERAN uses NHWC
        output = output.permute((0, 2, 3, 1))
        output = output.reshape((output_batches, -1))
        if is_np:
            if return_indices:
                return output.numpy(), indices.numpy()
            return output.numpy()
        if return_indices:
            return output, indices
        return output

    def from_indices(self, inputs, indices):
        """Computes the MaxPool given an input vector and indices of the maxes.

        This can be used to implement a Decoupled DNN, where you want to do
        effectively a MaxPool, but use the maxes computed based on a different
        input vector.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        indices = torch.tensor(indices, dtype=torch.long)

        # ERAN gives us NHWC
        inputs = self.window_data.unflatten_inputs(inputs)
        output_batches = inputs.shape[0]
        # PyTorch takes NCHW
        inputs = inputs.permute((0, 3, 1, 2))

        # https://discuss.pytorch.org/t/maxpool2d-indexing-order/8281/3
        inputs = torch.flatten(inputs, 2)
        output = torch.gather(inputs, 2, torch.flatten(indices, 2))
        output = output.view(indices.shape)

        output = output.permute((0, 2, 3, 1))
        output = output.reshape((output_batches, -1))

        if is_np:
            return output.numpy()
        return output

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        serialized.maxpool_data.window_data.CopyFrom(
            self.window_data.serialize())
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "maxpool_data":
            window_data = StridedWindowData.deserialize(
                serialized.maxpool_data.window_data)
            return cls(window_data)
        return None
