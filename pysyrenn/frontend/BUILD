py_library(
    name = "frontend",
    srcs = ["__init__.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":argmax_layer",
        ":averagepool_layer",
        ":concat_layer",
        ":conv2d_layer",
        ":fullyconnected_layer",
        ":hard_tanh_layer",
        ":maxpool_layer",
        ":network",
        ":normalize_layer",
        ":relu_layer",
        ":strided_window_data",
    ],
)

py_library(
    name = "transformer_client",
    srcs = ["transformer_client.py"],
    visibility = ["//pysyrenn/frontend:__subpackages__"],
    deps = [
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "network",
    srcs = ["network.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":averagepool_layer",
        ":concat_layer",
        ":conv2d_layer",
        ":fullyconnected_layer",
        ":hard_tanh_layer",
        ":maxpool_layer",
        ":normalize_layer",
        ":relu_layer",
        ":strided_window_data",
        ":transformer_client",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "layer",
    srcs = ["layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [],
)

py_library(
    name = "fullyconnected_layer",
    srcs = ["fullyconnected_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "relu_layer",
    srcs = ["relu_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "argmax_layer",
    srcs = ["argmax_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "hard_tanh_layer",
    srcs = ["hard_tanh_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "normalize_layer",
    srcs = ["normalize_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "strided_window_data",
    srcs = ["strided_window_data.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "conv2d_layer",
    srcs = ["conv2d_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        ":strided_window_data",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "maxpool_layer",
    srcs = ["maxpool_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        ":strided_window_data",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "averagepool_layer",
    srcs = ["averagepool_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        ":strided_window_data",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

# NOTE: .deserialize(...) requires you also include :network.
py_library(
    name = "concat_layer",
    srcs = ["concat_layer.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":layer",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)
