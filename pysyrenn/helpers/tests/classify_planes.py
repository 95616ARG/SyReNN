"""Tests the methods in classify_planes.py
"""
import numpy as np
import torch
import pytest
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend import Network, ReluLayer
from pysyrenn.helpers.classify_planes import PlanesClassifier

def test_compute_from_network():
    """Tests the it works given an arbitrary network and planes.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")
    network = Network([ReluLayer()])
    planes = [np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
              np.array([[3.0, 2.0], [7.0, 4.0], [5.0, 2.0]])]
    classifier = PlanesClassifier(network, planes, preimages=True)

    classifier.partial_compute()
    assert classifier.partially_computed
    transformed = network.transform_planes(planes, True, True)
    assert all(all(np.allclose(actual_polytope[0], truth_polytope[0]) and
                   np.allclose(actual_polytope[1], truth_polytope[1])
                   for actual_polytope, truth_polytope in zip(actual, truth))
               for actual, truth in zip(classifier.transformed_planes,
                                        transformed))

    classified = classifier.compute()
    assert len(classified) == len(planes)
    regions, labels = classified[0]
    assert np.allclose(regions, planes[0])
    assert np.allclose(labels, [1])
    regions, labels = classified[1]
    assert np.allclose(regions, planes[1])
    assert np.allclose(labels, [0])

    # Ensure it doesn't re-compute things it already knows.
    assert classifier.compute() is classified

def test_compute_from_syrenn():
    """Tests the it works given an arbitrary network and planes.
    """
    if not Network.has_connection():
        pytest.skip("No server connected.")

    network = Network([ReluLayer()])
    planes = [np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
              np.array([[3.0, 2.0], [7.0, 4.0], [5.0, 2.0]])]
    transformed = network.transform_planes(planes, True, True)

    classifier = PlanesClassifier.from_syrenn(transformed)
    assert classifier.partially_computed

    classified = classifier.compute()
    assert len(classified) == len(planes)
    regions, labels = classified[0]
    assert np.allclose(regions, planes[0])
    assert np.allclose(labels, [1])
    regions, labels = classified[1]
    assert np.allclose(regions, planes[1])
    assert np.allclose(labels, [0])

    # Ensure it doesn't re-compute things it already knows.
    assert classifier.compute() is classified

main(__name__, __file__)
