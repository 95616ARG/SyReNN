"""Tests the methods in classify_lines.py
"""
import numpy as np
import torch
import pytest
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend import Network, ReluLayer
from pysyrenn.helpers.integrated_gradients import IntegratedGradients

def test_compute_from_network():
    """Tests the it works given an arbitrary network and lines.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")
    network = Network([ReluLayer()])
    lines = [(np.array([0.0, 1.0]), np.array([0.0, -1.0])),
             (np.array([2.0, 3.0]), np.array([4.0, 3.0]))]

    helper = IntegratedGradients(network, lines)
    helper.partial_compute()
    assert len(helper.exactlines) == len(lines)
    assert np.allclose(helper.exactlines[0], [0.0, 0.5, 1.0])
    assert np.allclose(helper.exactlines[1], [0.0, 1.0])
    assert helper.n_samples == [2, 1]

    attributions_0 = helper.compute_attributions(0)
    assert len(attributions_0) == len(lines)
    # The second component doesn't affect the 0-label at all, and the first
    # component is 0 everywhere, so we have int_0^0 0.0dx = 0.0
    assert np.allclose(attributions_0[0], [0.0, 0.0])
    # Gradient of the 0-label is (1.0, 0.0) everywhere since its in the first
    # orthant, and the partition has a size of (2.0, 0.0), so the IG is (2.0,
    # 0.0).
    assert np.allclose(attributions_0[1], [2.0, 0.0])

    attributions_1 = helper.compute_attributions(1)
    assert len(attributions_1) == len(lines)
    # The gradient in the first partition is (0.0, 1.0) with a size of (0.0,
    # -1.0) -> contribution of (0.0, -1.0). In the second partition, (0.0,
    # 0.0)*(0.0, -1.0) = (0.0, 0.0).
    assert np.allclose(attributions_1[0], [0.0, -1.0])
    # Gradient is (0, 1) and the size is (2, 0) so IG is (0, 0).
    assert np.allclose(attributions_1[1], [0.0, 0.0])

    attributions_1_re = helper.compute_attributions(1)
    # Ensure it doesn't re-compute the attributions.
    assert attributions_1 is attributions_1_re

main(__name__, __file__)
