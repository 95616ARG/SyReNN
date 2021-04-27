"""Tests the methods in relu_layer.py
"""
from external.bazel_python.pytest_helper import main
from pysyrenn.frontend.layer import NetworkLayer

def test_stubs():
    """Tests that all of the methods are abstract/unimplemented.
    """
    layer = NetworkLayer()

    try:
        layer.compute(None)
        assert False
    except NotImplementedError:
        assert True

    try:
        layer.serialize()
        assert False
    except NotImplementedError:
        assert True

    try:
        layer.deserialize(None)
        assert False
    except NotImplementedError:
        assert True

main(__name__, __file__)
