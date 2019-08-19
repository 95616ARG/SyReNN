"""Methods describing layers in a neural network.
"""
class NetworkLayer:
    """Base class for layers that can operate on abstract VPolytopes.
    """
    def compute(self, inputs):
        """Transforms a concrete set of inputs.
        """
        raise NotImplementedError

    def serialize(self):
        """Serializes the layer for communication with the gRPC server.
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the layer from the Protobuf format.
        """
        raise NotImplementedError
