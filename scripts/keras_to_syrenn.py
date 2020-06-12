"""Script for converting sequential Keras models to SyReNN Networks.
"""
import tensorflow.keras as keras
import numpy as np
import pysyrenn

def keras_to_syrenn(model):
    """Converts a sequential Keras model to a SyReNN Network.

    Note that this conversion code makes a number of not-always-valid
    assumptions about the model; you should *always* manually verify that the
    returned SyReNN network has the same (within a small epsilon) output as the
    Keras model.
    """
    syrenn_layers = []
    def append_activation(function):
        """Adds activation function @function to the SyReNN layers.
        """
        if function is None or function is keras.activations.linear:
            # Identity: https://github.com/keras-team/keras/blob/bd024a1fc1cd6d88e8bc5da148968ff5e079caeb/keras/activations.py#L187
            pass
        elif function is keras.activations.relu:
            syrenn_layers.append(pysyrenn.ReluLayer())
        else:
            print(function)
            raise NotImplementedError

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        elif isinstance(layer, keras.layers.Conv2D):
            # filters: Height, Width, InChannels, OutChannels
            # biases: OutChannels
            filters, biases = map(to_numpy, layer.weights)
            if layer.padding == "same":
                pad_height = compute_same_padding(
                    filters.shape[0], layer.input_shape[1], layer.strides[0])
                pad_width = compute_same_padding(
                    filters.shape[1], layer.input_shape[2], layer.strides[1])

                assert pad_height % 2 == 0
                assert pad_width % 2 == 0

                padding = [pad_height // 2, pad_width // 2]
            elif layer.padding == "valid":
                padding = [0, 0]
            else:
                raise NotImplementedError

            window_data = pysyrenn.StridedWindowData(
                layer.input_shape[1:],  # HWC
                filters.shape[:2],  # HW
                layer.strides,  # HW
                padding,  # HW
                filters.shape[3])
     
            # Note that SyReNN *assumes* the HWIO format and transforms it
            # internally to the Pytorch OIHW format.
            syrenn_layers.append(
                pysyrenn.Conv2DLayer(window_data, filters, biases))
            append_activation(layer.activation)
        elif isinstance(layer, keras.layers.Activation):
            append_activation(layer.activation)
        elif isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, mean, var = map(to_numpy, layer.weights)
            # See https://github.com/keras-team/keras/blob/cb96315a291a8515544c6dd807500073958f8928/keras/backend/numpy_backend.py#L531
            # ((x - mean) / sqrt(var + epsilon)) * gamma + beta
            # = ((x - (mean - (d*beta))) / d) where
            # d := sqrt(var + epsilon) / gamma
            std = np.sqrt(var + 0.001) / gamma
            mean = mean - (std * beta)
            syrenn_layers.append(pysyrenn.NormalizeLayer(mean, std))
        elif isinstance(layer, keras.layers.MaxPooling2D):
            assert layer.padding == "valid"
            window_data = pysyrenn.StridedWindowData(
                layer.input_shape[1:],  # HWC
                layer.pool_size,  # HW
                layer.strides,  # HW
                [0, 0],  # HW
                layer.input_shape[3])
     
            # Note that SyReNN *assumes* the HWIO format and transforms it
            # internally to the Pytorch OIHW format.
            syrenn_layers.append(pysyrenn.MaxPoolLayer(window_data))
        elif isinstance(layer, keras.layers.Dropout):
            # Not needed for inference.
            pass
        elif isinstance(layer, keras.layers.Flatten):
            # By default, SyReNN passes data around in NHWC format to match with
            # ERAN/TF.
            assert layer.data_format == "channels_last"
        elif isinstance(layer, keras.layers.Dense):
            # weights: (from, to)
            # biases: (to,)
            weights, biases = map(to_numpy, layer.weights)
            syrenn_layers.append(pysyrenn.FullyConnectedLayer(weights, biases))
            append_activation(layer.activation)
        else:
            raise NotImplementedError
    return pysyrenn.Network(syrenn_layers)

def to_numpy(x):
    """Helper to convert TensorFlow tensors to Numpy.
    """
    return x.numpy()

def compute_same_padding(filter_size, in_size, stride):
    """Helper to compute the amount of padding used by a convolution.

    Computation based on https://stackoverflow.com/a/44242277
    """
    out_size = (in_size + (stride - 1)) // stride
    return max((out_size - 1) * stride + filter_size - in_size, 0)
