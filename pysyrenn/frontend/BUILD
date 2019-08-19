py_library(
    name = "frontend",
    srcs = ["__init__.py"],
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
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "transformer_client",
    srcs = ["transformer_client.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
    ],
    visibility = ["//pysyrenn/frontend:__subpackages__"],
)

py_library(
    name = "network",
    srcs = ["network.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":fullyconnected_layer",
        ":relu_layer",
        ":hard_tanh_layer",
        ":strided_window_data",
        ":conv2d_layer",
        ":maxpool_layer",
        ":averagepool_layer",
        ":normalize_layer",
        ":concat_layer",
        ":transformer_client",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "layer",
    srcs = ["layer.py"],
    deps = [],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "fullyconnected_layer",
    srcs = ["fullyconnected_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "relu_layer",
    srcs = ["relu_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "argmax_layer",
    srcs = ["argmax_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "hard_tanh_layer",
    srcs = ["hard_tanh_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "normalize_layer",
    srcs = ["normalize_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "strided_window_data",
    srcs = ["strided_window_data.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "conv2d_layer",
    srcs = ["conv2d_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
        ":strided_window_data",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "maxpool_layer",
    srcs = ["maxpool_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
        ":strided_window_data",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "averagepool_layer",
    srcs = ["averagepool_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
        ":strided_window_data",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "concat_layer",
    srcs = ["concat_layer.py"],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        ":layer",
    ],
    visibility = ["//:__subpackages__"],
)
