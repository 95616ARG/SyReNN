# PySyReNN Helpers
This directory contains a number of helper classes which utilize SyReNN to
compute desired properties of networks.

## List of Helpers
- [LinesClassifier](classify_lines.py) allows one to determine the output of a network on all
  points along a set of line segments.
- [PlanesClassifier](classify_planes.py) allows one to determine the output of a network on all
  points along a set of two-dimensional subspaces of the input space.
- [IntegratedGradients](integrated_gradients.py) supports exactly computing
  integrated gradients for a network and baseline/image pairs.

## Documentation
Primary usage examples are available in [../experiments](../experiments) along
with the test cases in [tests](tests).
