"""Methods for interacting with Masking Networks.
"""
import numpy as np
from pysyrenn.frontend.network import Network
from pysyrenn.frontend.fullyconnected_layer import FullyConnectedLayer
from pysyrenn.frontend.relu_layer import ReluLayer
import syrenn_proto.syrenn_pb2 as transformer_pb

class MaskingNetwork:
    """Implements a Masking Network.

    Currently supports:
    - Arbitrary Layers as long as the activation and values parameters are
      equal up to that layer.
    - Once the activation and values parameters differ, only FullyConnected and
      ReLU layers are supported (note that other layers can be supported, but
      it would require extending the .compute method and are not needed for the
      ACAS networks).
    """
    def __init__(self, activation_layers, value_layers):
        """Constructs the new PatchedNetwork.

        @activation_layers is a list of layers defining the values of the
            activation vectors.
        @value_layers is a list of layers defining the values of the value
            vectors.

        Note that:
        1. To match the notation in the paper, you should consider the ith
            activation layer and the ith value layer being ``intertwined`` into
            a single layer that produces an (activation, value) tuple.
        2. To that end, the number, types, and output sizes of the layers in
            @activation_layers and @values_layers should match.
        3. ReluLayers are interpreted as MReLU layers.
        """
        self.activation_layers = activation_layers
        self.value_layers = value_layers

        self.n_layers = len(activation_layers)
        assert self.n_layers == len(value_layers)

        try:
            self.differ_index = next(
                l for l in range(self.n_layers)
                if activation_layers[l] is not value_layers[l]
            )
        except StopIteration:
            self.differ_index = len(value_layers)

    def compute(self, inputs):
        """Computes the output of the Masking Network on @inputs.

        @inputs should be a Numpy array of inputs.
        """
        # Up to differ_index, the values and activation vectors are the same.
        pre_network = Network(self.activation_layers[:self.differ_index])
        mid_inputs = pre_network.compute(inputs)
        # Now we have to actually separately handle the masking when
        # activations != values.
        activation_vector = mid_inputs
        value_vector = mid_inputs
        for layer_index in range(self.differ_index, self.n_layers):
            activation_layer = self.activation_layers[layer_index]
            value_layer = self.value_layers[layer_index]
            if isinstance(activation_layer, FullyConnectedLayer):
                activation_vector = activation_layer.compute(activation_vector)
                value_vector = value_layer.compute(value_vector)
            elif isinstance(activation_layer, ReluLayer):
                mask = np.maximum(np.sign(activation_vector), 0.0)
                value_vector *= mask
                activation_vector *= mask
            else:
                raise NotImplementedError
        return value_vector

    def serialize(self):
        """Serializes the MaskingNetwork to the Protobuf format.

        Notably, the value_net only includes layers after differ_index.
        """
        serialized = transformer_pb.MaskingNetwork()
        serialized.activation_layers.extend([
            layer.serialize() for layer in self.activation_layers
        ])
        serialized.value_layers.extend([
            layer.serialize()
            for layer in self.value_layers[self.differ_index:]
        ])
        serialized.differ_index = self.differ_index
        return serialized

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the MaskingNetwork from the Protobuf format.
        """
        activation_layers = serialized.activation_layers
        activation_layers = Network.deserialize_layers(activation_layers)

        value_layers = serialized.value_layers
        value_layers = Network.deserialize_layers(value_layers)

        differ_index = serialized.differ_index
        value_layers = activation_layers[:differ_index] + value_layers
        return cls(activation_layers, value_layers)
