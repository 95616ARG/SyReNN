"""Tests the methods in classify_lines.py
"""
import numpy as np
import torch
import pytest
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend import Network, ReluLayer
from pysyrenn.helpers.classify_lines import LinesClassifier

def test_compute_from_network():
    """Tests the it works given an arbitrary network and lines.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")

    network = Network([ReluLayer()])
    lines = [(np.array([0.0, 1.0]), np.array([0.0, -1.0])),
             (np.array([2.0, 3.0]), np.array([4.0, 3.0]))]
    classifier = LinesClassifier(network, lines, preimages=True)

    classifier.partial_compute()
    exactlines = network.exactlines(lines, True, True)
    assert all(np.allclose(actual[0], truth[0]) and
               np.allclose(actual[1], truth[1])
               for actual, truth in zip(classifier.transformed_lines, exactlines))

    classified = classifier.compute()
    assert len(classified) == len(lines)
    regions, labels = classified[0]
    assert np.allclose(regions, [[[0.0, 1.0], [0.0, 0.0]],
                                 [[0.0, 0.0], [0.0, -1.0]]])
    assert np.allclose(labels, [1, 0])
    regions, labels = classified[1]
    assert np.allclose(regions, [[[2.0, 3.0], [3.0, 3.0]],
                                 [[3.0, 3.0], [4.0, 3.0]]])
    assert np.allclose(labels, [1, 0])

    # Ensure it doesn't re-compute things it already knows.
    assert classifier.compute() is classified

def test_compute_from_exactlines():
    """Tests the it works given pre-transformed lines.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")

    network = Network([ReluLayer()])
    lines = [(np.array([0.0, 1.0]), np.array([0.0, -1.0])),
             (np.array([2.0, 3.0]), np.array([4.0, 3.0]))]
    exactlines = network.exactlines(lines, True, True)

    classifier = LinesClassifier.from_exactlines(exactlines)
    classified = classifier.compute()

    assert len(classified) == len(lines)
    regions, labels = classified[0]
    assert np.allclose(regions, [[[0.0, 1.0], [0.0, 0.0]],
                                 [[0.0, 0.0], [0.0, -1.0]]])
    assert np.allclose(labels, [1, 0])
    regions, labels = classified[1]
    assert np.allclose(regions, [[[2.0, 3.0], [3.0, 3.0]],
                                 [[3.0, 3.0], [4.0, 3.0]]])
    assert np.allclose(labels, [1, 0])

def test_compute_from_exactline_error():
    """Tests that it requires the plural exactline*s*(), not the singular.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")

    network = Network([ReluLayer()])
    exactline = network.exactline([-1.0, 1.0], [1.0, 0.0], True, True)

    try:
        classifier = LinesClassifier.from_exactlines(exactline)
        assert False
    except TypeError as e:
        pass

main(__name__, __file__)
