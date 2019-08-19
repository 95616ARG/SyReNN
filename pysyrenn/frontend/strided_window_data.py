"""Methods for handling functions that compute strided windows over images.
"""
import syrenn_proto.syrenn_pb2 as transformer_pb

class StridedWindowData:
    """Represents metadata about a function that computes strided windows.

    Used by Conv2DLayer, MaxPoolLayer, and AveragePoolLayer.
    """
    def __init__(self, input_shape, window_shape, strides,
                 padding, out_channels):
        """Initializes a new StridedWindowData.

        input_shape = (height, width, channels)
        window_shape = (height, width)
        strides = (height, width)
        padding = (height, width)
        out_channels = #
        """
        self.input_shape = tuple(input_shape)
        self.window_shape = tuple(window_shape)
        self.strides = tuple(strides)
        self.padding = tuple(padding)
        self.out_channels = out_channels

    def out_height(self):
        """Returns the height dimension of the layer's output.
        """
        return ((
            ((2 * self.padding[0]) + self.input_shape[0] - self.window_shape[0])
            // self.strides[0]) + 1)

    def out_width(self):
        """Returns the width dimension of the layer's output.
        """
        return ((
            ((2 * self.padding[1]) + self.input_shape[1] - self.window_shape[1])
            // self.strides[1]) + 1)

    def out_shape(self):
        """Returns the output shape of the corresponding layer (H, W, C).
        """
        return (self.out_height(), self.out_width(), self.out_channels)

    def unflatten_inputs(self, inputs):
        """Returns a shaped version of the flattened inputs.
        """
        return inputs.reshape((-1,) + self.input_shape)

    def serialize(self):
        """Serializes the StridedWindowData for the transformer server.
        """
        serialized = transformer_pb.StridedWindowData()

        serialized.in_height = self.input_shape[0]
        serialized.in_width = self.input_shape[1]
        serialized.in_channels = self.input_shape[2]

        serialized.window_height = self.window_shape[0]
        serialized.window_width = self.window_shape[1]
        serialized.out_channels = self.out_channels

        serialized.stride_height = self.strides[0]
        serialized.stride_width = self.strides[1]

        serialized.pad_height = self.padding[0]
        serialized.pad_width = self.padding[1]

        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the StridedWindowData from the Protobuf format.
        """
        in_shape = (serialized.in_height, serialized.in_width,
                    serialized.in_channels)
        window_shape = (serialized.window_height, serialized.window_width)
        strides = (serialized.stride_height, serialized.stride_width)
        pad = (serialized.pad_height, serialized.pad_width)
        out_channels = serialized.out_channels
        return cls(in_shape, window_shape, strides, pad, out_channels)
