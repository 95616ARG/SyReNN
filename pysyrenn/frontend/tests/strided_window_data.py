"""Tests the methods in strided_window_data.py
"""
import numpy as np
import torch
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.strided_window_data import StridedWindowData

def test_out_shape():
    """Tests that the out_* methods work correctly.
    """
    in_shape = (32, 64, np.random.randint(1, 128))
    out_channels = np.random.randint(1, 128)
    window_shape = (4, 2)
    strides = (3, 1)
    pad = (1, 3)
    window_data = StridedWindowData(in_shape, window_shape, strides,
                                    pad, out_channels)
    # After padding, height is 34. 
    # [0 - 4), [3 - 7), [6 - 10), ..., [30 - 34)
    assert window_data.out_height() == 11
    # After padding, width is 70. 
    # [0 - 2), [1 - 3), [2 - 4), ..., [68 - 70)
    assert window_data.out_width() == 69
    assert window_data.out_shape() == (11, 69, out_channels)

def test_serialize():
    """Tests that it correctly [de]serializes.
    """
    in_shape = (32, 64, np.random.randint(1, 128))
    out_channels = np.random.randint(1, 128)
    window_shape = (4, 2)
    strides = (3, 1)
    pad = (1, 3)
    window_data = StridedWindowData(in_shape, window_shape, strides,
                                    pad, out_channels)
    serialized = window_data.serialize()
    assert serialized.in_height == 32
    assert serialized.in_width == 64
    assert serialized.in_channels == in_shape[2]

    assert serialized.window_height == 4
    assert serialized.window_width == 2
    assert serialized.out_channels == out_channels

    assert serialized.stride_height == 3
    assert serialized.stride_width == 1

    assert serialized.pad_height == 1
    assert serialized.pad_width == 3

    assert StridedWindowData.deserialize(serialized).serialize() == serialized

main(__name__, __file__)
