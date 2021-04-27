"""Methods describing a 2D Convolution layer.
"""
import numpy as np
import torch
from pysyrenn.frontend.layer import NetworkLayer
from pysyrenn.frontend.strided_window_data import StridedWindowData
import syrenn_proto.syrenn_pb2 as transformer_pb

class Conv2DLayer(NetworkLayer):
    """Represents a 2D Convolution layer in a network.
    """
    def __init__(self, window_data, filter_weights, biases):
        """Constructs a new Conv2DLayer.
        """
        self.window_data = window_data
        self.filter_weights = torch.tensor(filter_weights, dtype=torch.float32)
        self.biases = torch.tensor(biases, dtype=torch.float32)

    def compute(self, inputs, jacobian=False):
        """Computes the 2D convolution given an input vector.

        If @jacobian=True, it only computes the homogeneous portion (i.e., does
        not add biases). This can be used to compute the product of the
        Jacobian of the layer with its inputs.
        """
        is_np = isinstance(inputs, np.ndarray)
        if is_np:
            inputs = torch.tensor(inputs, dtype=torch.float32)
        # ERAN gives us NHWC
        inputs = self.window_data.unflatten_inputs(inputs)
        output_batches = inputs.shape[0]
        # PyTorch takes NCHW
        inputs = inputs.permute((0, 3, 1, 2))

        # ERAN gives us HWIcOc, but Pytorch takes OcIcHW
        filters = self.filter_weights.permute((3, 2, 0, 1))
        biases = self.biases
        if jacobian:
            biases = torch.zeros_like(biases)

        output = torch.nn.functional.conv2d(inputs, filters, biases,
                                            self.window_data.strides,
                                            self.window_data.padding)
        # Pytorch gives us NCHW, ERAN uses NHWC
        output = output.permute((0, 2, 3, 1))
        output = output.reshape((output_batches, -1))
        if is_np:
            return output.detach().numpy()
        return output

    def serialize(self):
        """Serializes the layer for use with the transformer server.
        """
        serialized = transformer_pb.Layer()
        filters = self.filter_weights.numpy()
        biases = self.biases.numpy()

        conv2d_data = serialized.conv2d_data

        conv2d_data.window_data.CopyFrom(self.window_data.serialize())
        conv2d_data.filters.extend(list(filters.flatten()))
        serialized.conv2d_data.biases.extend(list(biases.flatten()))
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes from the Protobuf format.
        """
        if serialized.WhichOneof("layer_data") == "conv2d_data":
            window_data = StridedWindowData.deserialize(
                serialized.conv2d_data.window_data)
            filters = np.array(serialized.conv2d_data.filters)
            filters = filters.reshape(window_data.window_shape +
                                      (window_data.input_shape[2],
                                       window_data.out_channels,))
            biases = np.array(serialized.conv2d_data.biases)
            return cls(window_data, filters, biases)
        return None
