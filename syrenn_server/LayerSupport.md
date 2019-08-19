# Supported Layers
We support the following layers (ones that are used in our experiments, and thus
somewhat better-tested, are marked in **bold**):

- **ArgMax**
    - Post-vertices should not be used, as ArgMax is undefined on the
    endpoints of linear regions.
- AveragePool
    - Zeros from padding _are_ included when computing averages. This can be
      changed by modifying the MKL-DNN call in
      [averagepool_transformer.cc](averagepool_transformer.cc).
- Concat
    - Concatenation along flattened dimensions and channel dimensions is
      supported, however currently only Conv2D layers can be concatenated
      across channel dimensions.
    - Concat currently only supports a "depth" of one; if you need to
      concat "stacks" of layers, you will need to modify the transformer
      to support this (or refactor your model to avoid it).
- **Conv2D**
- **FullyConnected**
- **HardTanh**
- MaxPool
    - Only ExactLine (i.e. one-dimensional restriction domains of interest)  is currently supported.
    - Zeros from padding are _never_ used as maximum values.
- **Normalize**
- ReLU + MaxPool fused (automatically optimized by the server).
- **ReLU**

### Supporting New Layers
If you want to implement initial support for a new layer, you have two main
options:

1. If the layer is affine, sub-class ``AffineTransformer`` (see
   ``normalize_transformer.cc`` for an example).
2. If the layer is piecewise-linear, sub-class ``PWLTransformer`` (see
   ``relu_transformer.cc`` for an example).

Sub-classing from these avoids explicitly writing transformer code, instead the
layer function itself is described relatively-abstractly and then used by the
transformer implementations in the base classes.

Please note that ``PWLTransformer`` may not be an optimal implementation for
all piecewise-linear functions, so you may get better performance by writing
your own implementations of the transformer instead of using the defaults
provided by ``PWLTransformer``.

If you have issues, feel free to contact us for support.