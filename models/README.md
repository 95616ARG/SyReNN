# Models Used
This directory contains code for generating and referencing the models used in
[../experiments](../experiments). In particular, we use models from:

1. [ERAN](https://github.com/eth-sri/eran)
2. [ReluPlex](https://github.com/guykatzz/ReluplexCav2017)
3. [VRL](https://github.com/caffett/VRL_CodeReview)
4. [ONNX](https://github.com/onnx/models)

In general, we prefer the ERAN format for its simplicity, and translate the
ReluPlex and VRL models to it before importing in our experiments. The ONNX
models are not used in our experiments, but we use them to test the
(experimental) ONNX import support in PySyReNN.