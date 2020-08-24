py_library(
    name = "helpers",
    srcs = ["helpers.py"],
    visibility = ["//:__subpackages__"],
)

py_test(
    name = "relu_layer",
    size = "small",
    srcs = ["relu_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:relu_layer",
    ],
)

py_test(
    name = "hard_tanh_layer",
    size = "small",
    srcs = ["hard_tanh_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:hard_tanh_layer",
    ],
)

py_test(
    name = "argmax_layer",
    size = "small",
    srcs = ["argmax_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:argmax_layer",
    ],
)

py_test(
    name = "fullyconnected_layer",
    size = "small",
    srcs = ["fullyconnected_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:fullyconnected_layer",
    ],
)

py_test(
    name = "normalize_layer",
    size = "small",
    srcs = ["normalize_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:normalize_layer",
    ],
)

py_test(
    name = "averagepool_layer",
    size = "small",
    srcs = ["averagepool_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:averagepool_layer",
        "//pysyrenn/frontend:strided_window_data",
    ],
)

py_test(
    name = "maxpool_layer",
    size = "small",
    srcs = ["maxpool_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:maxpool_layer",
        "//pysyrenn/frontend:strided_window_data",
    ],
)

py_test(
    name = "conv2d_layer",
    size = "small",
    srcs = ["conv2d_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:conv2d_layer",
        "//pysyrenn/frontend:strided_window_data",
    ],
)

py_test(
    name = "concat_layer",
    size = "small",
    srcs = ["concat_layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:averagepool_layer",
        "//pysyrenn/frontend:concat_layer",
        "//pysyrenn/frontend:conv2d_layer",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:normalize_layer",
        "//pysyrenn/frontend:strided_window_data",
    ],
)

py_test(
    name = "strided_window_data",
    size = "small",
    srcs = ["strided_window_data.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:strided_window_data",
    ],
)

py_test(
    name = "network",
    size = "medium",
    srcs = ["network.py"],
    data = ["@onnx_squeezenet//:all"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:argmax_layer",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:relu_layer",
        "//pysyrenn/frontend:transformer_client",
    ],
)

py_test(
    name = "layer",
    size = "small",
    srcs = ["layer.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:layer",
    ],
)

py_test(
    name = "transformer_client",
    size = "small",
    srcs = ["transformer_client.py"],
    deps = [
        ":helpers",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:relu_layer",
        "//pysyrenn/frontend:transformer_client",
    ],
)
