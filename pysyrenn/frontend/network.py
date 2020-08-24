"""
This is modified from the read_net_file.py file in ERAN. It supports a
simplified interface to reading the network files, as long as they are
sequential (i.e., it assumes each layer's input is only the previous layer's
output).

This is the main abstraction "orchestrator" --- to check robustness around a
point, for example, you would create a Network instance and call
class_on_input, which would then call the transform_polytopes methods on each
layer to get the output state.

Not all layer types are supported, but this should be enough to run the
feed-forward MNIST networks. We can add more complexity if desired later.
"""
import ast
import numpy as np
import onnx
from onnx import numpy_helper
from onnx import shape_inference
import torch
import syrenn_proto.syrenn_pb2 as transformer_pb
from pysyrenn.frontend import transformer_client
from pysyrenn.frontend.fullyconnected_layer import FullyConnectedLayer
from pysyrenn.frontend.relu_layer import ReluLayer
from pysyrenn.frontend.hard_tanh_layer import HardTanhLayer
from pysyrenn.frontend.normalize_layer import NormalizeLayer
from pysyrenn.frontend.strided_window_data import StridedWindowData
from pysyrenn.frontend.conv2d_layer import Conv2DLayer
from pysyrenn.frontend.maxpool_layer import MaxPoolLayer
from pysyrenn.frontend.averagepool_layer import AveragePoolLayer
from pysyrenn.frontend.concat_layer import ConcatLayer

LAYER_TYPES = [FullyConnectedLayer, ReluLayer, HardTanhLayer, NormalizeLayer,
               Conv2DLayer, MaxPoolLayer, AveragePoolLayer, ConcatLayer]

class Network:
    """Represents a sequential, feed-forward network.
    """
    def __init__(self, layers):
        """Initializes a network given a list of layers.
        """
        self.layers = layers

    @classmethod
    def has_connection(cls):
        """Returns True iff the transformer server can be reached.
        """
        try:
            network = cls([ReluLayer()])
            network.exactline([0.0], [1.0], False, False)
            return True
        except Exception as exception:
            if "failed to connect" in exception.details():
                return False
            raise exception

    def compute(self, inputs):
        """Returns the concrete transformation of the inputs.

        inputs may be a Numpy array or PyTorch Tensor.
        """
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        if isinstance(inputs, np.ndarray) and len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        outputs = inputs
        for layer in self.layers:
            outputs = layer.compute(outputs)
        return outputs

    def compute_gradients(self, inputs, label):
        """Computes gradients of the network at each input point.

        We use PyTorch as it's a big pain to get TensorFlow to import with
        Bazel. It's also (arguably) a nicer interface for this kind of thing.
        """
        input_tensor = torch.tensor(inputs, dtype=torch.float32,
                                    requires_grad=True)
        output = torch.sum(self.compute(input_tensor)[:, label])
        output.backward()
        return input_tensor.grad.detach().numpy()

    def serialize(self):
        """Serializes the network into the Protobuf format.

        NOTE: This should be used for serializing to disk, *NOT* for sending to
        the transformer server. gRPC does not support very large message sizes,
        so requests to the transformer server should send each layer
        individually.
        """
        serialized = transformer_pb.Network()
        serialized.layers.extend([layer.serialize() for layer in self.layers])
        return serialized

    @staticmethod
    def deserialize_layers(serialized):
        """Helper method to deserialize a list of layers from Protobuf format.

        Used for deserializing MaskingNetworks in ../helpers/masking_network.py
        and for deserializing Networks in .deserialize below.
        """
        layers = []
        for layer in serialized:
            any_matched = False
            for layer_type in LAYER_TYPES:
                deserialized = layer_type.deserialize(layer)
                if deserialized:
                    layers.append(deserialized)
                    any_matched = True
                    break
            if not any_matched:
                raise NotImplementedError
        return layers

    @classmethod
    def deserialize(cls, serialized):
        """Deserializes the network from the Protobuf format.
        """
        return cls(cls.deserialize_layers(serialized.layers))

    def exactlines(self, lines, compute_preimages, include_post):
        """Computes ExactLine for this network restricted to a set of lines.

        If @include_post=False, a list is returned of the endpoints.
        If @include_post=True, a list is returned of tuples (endpoints,
        network(endpoints)).
        """
        lines = np.asarray(lines)
        linewise_results = transformer_client.transform_lines(
            self, lines, include_post)
        for i, (start, end) in enumerate(lines):
            endpoint_ratios, post_vertices = linewise_results[i]
            if compute_preimages:
                preimages = start + np.outer(endpoint_ratios, end - start)
                linewise_results[i] = (preimages, post_vertices)
            if not include_post:
                linewise_results[i] = linewise_results[i][0]
        return linewise_results

    def exactline(self, start, end, compute_preimages, include_post):
        """Helper method for computing ExactLine on a single line.

        Identical to ExactLines(), except it takes only a single line and
        returns only a single transformed line. See exactlines above.
        """
        lines = [(start, end)]
        return self.exactlines(lines, compute_preimages, include_post)[0]

    def transform_planes(self, planes, compute_preimages, include_post):
        """Computes the symbolic representation for each of a set of planes.

        Returns a list of UPolytopes (one per plane in @planes), which are
        themselves lists of either tuples of Numpy arrays (pre, post) or just
        Numpy arrays pre depending on @include_post. Note that
        @compute_preimages=False still returns (pre, post), except the @pre
        matrices are *combinations* of the original vertices, not points
        themselves.
        """
        planes = np.asarray(planes)
        transformed = transformer_client.transform_planes(self, planes)
        if compute_preimages:
            for i, input_plane in enumerate(planes):
                upolytope = transformed[i]
                for j, vpolytope in enumerate(upolytope):
                    combinations, postimages = vpolytope
                    # (n_post, n_pre).(n_pre, n_dims)
                    preimages = np.matmul(combinations, input_plane)
                    transformed[i][j] = (preimages, postimages)
        if not include_post:
            for i, vpolytope in enumerate(transformed):
                for j, (pre, post) in enumerate(vpolytope):
                    vpolytope[j] = pre
        return transformed

    def transform_plane(self, plane, compute_preimages, include_post):
        """Computes the symbolic representation for a single plane.

        Identical to @transform_planes, except it takes only a single plane and
        returns only a single UPolytope (instead of a list of them).
        """
        return self.transform_planes([plane], compute_preimages,
                                     include_post)[0]

    @classmethod
    def parse_np_array(cls, serialized):
        """Given a string, returns a Numpy array of its contents.

        Used when parsing the ERAN model definition files.
        """
        if isinstance(serialized, str):
            return np.array(ast.literal_eval(serialized))
        # Helper to read directly from a file.
        return cls.parse_np_array(serialized.readline()[:-1].strip())

    @classmethod
    def from_file(cls, net_file, file_type=None):
        """Loads a network from an ONNX or ERAN file format.

        Files ending in .onnx will be loaded as ONNX files, ones ending in
        .eran will be loaded as ERAN files. Pass file_tye="{eran, onnx}" to
        override this behavior.
        """
        if file_type is None:
            file_type = net_file.split(".")[-1]
        file_type = file_type.lower()

        if file_type == "eran":
            return cls.from_eran(net_file)
        if file_type == "onnx":
            return cls.from_onnx(net_file)
        raise NotImplementedError

    @classmethod
    def from_eran(cls, net_file):
        """Helper method to read an ERAN net_file into a Network.

        Currently only supports a subset of those supported by the original
        read_net_file.py. See an example of the type of network file we're
        reading here:

        https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf

        This code has been adapted (with heavy modifications) from the ERAN
        source code. Each layer has a header line that describes the type of
        layer, which is then followed by the weights (if applicable). Note that
        some layers are rolled together in ERAN but we do separately (eg.
        "ReLU" in the ERAN format corresponds to Affine + ReLU in our
        representation).
        """
        layers = []
        net_file = open(net_file, "r")
        while True:
            curr_line = net_file.readline()[:-1]
            if curr_line in {"Affine", "ReLU", "HardTanh"}:
                # Parses a fully-connected layer, possibly followed by
                # non-linearity.
                # ERAN files use (out_dims, in_dims), we use the opposite.
                weights = cls.parse_np_array(net_file).transpose()
                biases = cls.parse_np_array(net_file)

                if len(layers) > 1 and isinstance(layers[-2], Conv2DLayer):
                    # When there's an affine after a 2D convolution, ERAN's
                    # files assume the input is CHW when it's actually HWC. We
                    # correct that here by permuting the dimensions.
                    conv_layer = layers[-2]
                    output_size = weights.shape[-1]
                    weights = weights.reshape(
                        (conv_layer.window_data.out_channels,
                         conv_layer.window_data.out_height(),
                         conv_layer.window_data.out_width(),
                         output_size))
                    weights = weights.transpose(1, 2, 0, 3)
                    weights = weights.reshape((-1, output_size))

                # Add the fully-connected layer.
                layers.append(FullyConnectedLayer(weights, biases))

                # Maybe add a non-linearity.
                if curr_line == "ReLU":
                    layers.append(ReluLayer())
                elif curr_line == "HardTanh":
                    layers.append(HardTanhLayer())
            elif curr_line.startswith("Normalize"):
                # Parses a Normalize layer.
                means = curr_line.split("mean=")[1].split("std=")[0].strip()
                means = cls.parse_np_array(means)

                stds = curr_line.split("std=")[1].strip()
                stds = cls.parse_np_array(stds)

                layers.append(NormalizeLayer(means, stds))
            elif curr_line.startswith("Conv2D"):
                # Parses a 2D-Convolution layer. The info line looks like:
                # ReLU, filters=16, kernel_size=[4, 4], \
                # input_shape=[28, 28, 1], stride=[2, 2], padding=0
                # But, we can get filters and kernel_size from the actual
                # filter weights, so no need to parse that here.
                info_line = net_file.readline()[:-1].strip()
                activation = info_line.split(",")[0]

                stride = cls.parse_np_array(
                    info_line.split("stride=")[1].split("],")[0] + "]")

                input_shape = info_line.split("input_shape=")[1].split("],")[0]
                input_shape += "]"
                input_shape = cls.parse_np_array(input_shape)

                pad = (0, 0)
                if "padding=" in info_line:
                    pad = int(info_line.split("padding=")[1])
                    pad = (pad, pad)

                # (f_h, f_w, i_c, o_c)
                filter_weights = cls.parse_np_array(net_file)
                # (o_c,)
                biases = cls.parse_np_array(net_file)

                window_data = StridedWindowData(
                    input_shape, filter_weights.shape[:2],
                    stride, pad, filter_weights.shape[3])
                layers.append(Conv2DLayer(window_data, filter_weights, biases))

                if activation == "ReLU":
                    layers.append(ReluLayer())
                elif activation == "HardTanh":
                    layers.append(HardTanhLayer())
                else:
                    # As far as I know, all Conv2D layers should have an
                    # associated activation function in the ERAN format.
                    raise NotImplementedError
            elif curr_line.strip() == "":
                break
            else:
                raise NotImplementedError
        return cls(layers)

    @staticmethod
    def onnx_ints_attribute(node, name):
        """Reads int attributes (eg. weight shape) from an ONNX node.
        """
        return next(attribute.ints
                    for attribute in node.attribute
                    if attribute.name == name)

    @staticmethod
    def layer_from_onnx(graph, node):
        """Reads a layer from an ONNX node.

        Specs for the ONNX operators are available at:
        https://github.com/onnx/onnx/blob/master/docs/Operators.md
        """
        # First, we get info about inputs to the layer (including previous
        # layer outputs & things like weight matrices).
        inputs = node.input
        deserialized_inputs = []
        deserialized_input_shapes = []
        for input_name in inputs:
            # We need to find the initializers (which I think are basically
            # weight tensors) for the particular input.
            initializers = [init for init in graph.initializer
                            if str(init.name) == str(input_name)]
            if initializers:
                assert len(initializers) == 1
                # Get the weight tensor as a Numpy array and save it.
                deserialized_inputs.append(numpy_helper.to_array(initializers[0]))
            else:
                # This input is the output of another node, so just store the
                # name of that other node (we'll link them up later). Eg.
                # squeezenet0_conv0_fwd.
                deserialized_inputs.append(str(input_name))
            # Get metadata about the input (eg. its shape).
            infos = [info for info in graph.value_info
                     if info.name == input_name]
            if infos:
                # This is an input with a particular shape.
                assert len(infos) == 1
                input_shape = [d.dim_value
                               for d in infos[0].type.tensor_type.shape.dim]
                deserialized_input_shapes.append(input_shape)
            elif input_name == "data":
                # This is an input to the entire network, its handled
                # separately.
                net_input_shape = graph.input[0].type.tensor_type.shape
                input_shape = [d.dim_value for d in net_input_shape.dim]
                deserialized_input_shapes.append(input_shape)
            else:
                # This doesn't have any inputs.
                deserialized_input_shapes.append(None)

        layer = None

        # Standardize some of the data shared by the strided-window layers.
        if node.op_type in {"Conv", "MaxPool", "AveragePool"}:
            # NCHW -> NHWC
            input_shape = deserialized_input_shapes[0]
            input_shape = [input_shape[2], input_shape[3], input_shape[1]]
            strides = list(Network.onnx_ints_attribute(node, "strides"))
            pads = list(Network.onnx_ints_attribute(node, "pads"))
            # We do not support separate begin/end padding.
            assert pads[0] == pads[2]
            assert pads[1] == pads[3]
            pads = pads[1:3]

        # Now, parse the actual layers.
        if node.op_type == "Conv":
            # We don't support dilations or non-1 groups.
            dilations = list(Network.onnx_ints_attribute(node, "dilations"))
            assert all(dilation == 1 for dilation in dilations)
            group = Network.onnx_ints_attribute(node, "group")
            assert not group or group == 1

            # biases are technically optional, but I don't *think* anyone uses
            # that feature.
            assert len(deserialized_inputs) == 3
            input_data, filters, biases = deserialized_inputs
            # OIHW -> HWIO
            filters = filters.transpose((2, 3, 1, 0))

            window_data = StridedWindowData(input_shape, filters.shape[:2],
                                            strides, pads, biases.shape[0])
            layer = Conv2DLayer(window_data, filters, biases)
        elif node.op_type == "Relu":
            layer = ReluLayer()
        elif node.op_type == "MaxPool":
            kernel_shape = Network.onnx_ints_attribute(node, "kernel_shape")
            window_data = StridedWindowData(input_shape, list(kernel_shape),
                                            strides, pads, input_shape[2])
            layer = MaxPoolLayer(window_data)
        elif node.op_type == "AveragePool":
            kernel_shape = Network.onnx_ints_attribute(node, "kernel_shape")
            window_data = StridedWindowData(input_shape, list(kernel_shape),
                                            strides, pads, input_shape[2])
            layer = AveragePoolLayer(window_data)
        elif node.op_type == "Gemm":
            input_data, weights, biases = deserialized_inputs

            alpha = Network.onnx_ints_attribute(node, "alpha")
            if alpha:
                weights *= alpha
            beta = Network.onnx_ints_attribute(node, "beta")
            if beta:
                biases *= beta

            trans_A = Network.onnx_ints_attribute(node, "transA")
            trans_B = Network.onnx_ints_attribute(node, "transB")

            # We compute (X . W) [+ C].
            assert not trans_A
            if trans_B:
                weights = weights.transpose()
            layer = FullyConnectedLayer(weights, biases)
        elif node.op_type == "BatchNormalization":
            epsilon = Network.onnx_ints_attribute(node, "epsilon")
            input_data, scale, B, mean, var = deserialized_inputs
            # We don't yet support separate scale/bias parameters, though they
            # can be rolled in to mean/var.
            assert np.allclose(scale, 1.0) and np.allclose(B, 0.0)
            layer = NormalizeLayer(mean, np.sqrt(var + epsilon))
        elif node.op_type == "Concat":
            layer = list(inputs)
        elif node.op_type in {"Dropout", "Reshape", "Flatten"}:
            # These are (more-or-less) handled implicitly since we pass around
            # flattened activation vectors and only work with testing.
            layer = False
        else:
            raise NotImplementedError
        assert len(node.output) == 1
        return (inputs[0], node.output[0], layer)

    @classmethod
    def from_onnx(cls, net_file):
        """Reads a network from an ONNX file.
        """
        model = onnx.load(net_file)
        model = shape_inference.infer_shapes(model)
        # layers will be {output_name: layer}
        layers = {}
        # First, we just convert everything we can into a layer
        for node in model.graph.node:
            layer = cls.layer_from_onnx(model.graph, node)
            if layer is not False:
                input_name, output_name, layer = layer
                layers[output_name] = (input_name, layer)

        # Then, we roll all of the concat layers together
        while any(l for l in layers.values() if isinstance(l[1], list)):
            concat_name = next(name for name, layer in layers.items()
                               if isinstance(layer[1], list))
            _, input_layers = layers[concat_name]
            assert all(isinstance(layers[input_name][1], ReluLayer)
                       for input_name in input_layers)
            # We move the relus to behind the concat
            relu_layer_names = input_layers.copy()
            input_layer_names = [layers[input_name][0]
                                 for input_name in relu_layer_names]
            entire_input_name = layers[input_layer_names[0]][0]
            assert all(entire_input_name == layers[input_layer][0]
                       for input_layer in input_layer_names)
            input_layers = [layers[input_name][1]
                            for input_name in input_layer_names]
            for input_layer_name in input_layer_names:
                layers.pop(input_layer_name)
            # Then we remove all of the intermediate relu layers
            for relu_layer_name in relu_layer_names:
                layers.pop(relu_layer_name)
            concat_layer = ConcatLayer(input_layers)
            layers[concat_name + "_prerelu"] = (entire_input_name, concat_layer)
            layers[concat_name] = (concat_name + "_prerelu", ReluLayer())

        # Then, we flatten
        flat_layers = []
        input_name = "data"
        while layers:
            next_input_name, next_layer = next(
                (output_name, layer[1])
                for output_name, layer in layers.items()
                if layer[0] == input_name)
            flat_layers.append(next_layer)
            input_name = next_input_name
            layers.pop(next_input_name)
        flat_layers = [layer for layer in flat_layers if layer is not False]
        return cls(flat_layers)
