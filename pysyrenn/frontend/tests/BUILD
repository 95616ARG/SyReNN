py_test(
    name = "relu_layer",
    size = "small",
    srcs = ["relu_layer.py"],
    deps = [
        "//pysyrenn/frontend:relu_layer",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "hard_tanh_layer",
    size = "small",
    srcs = ["hard_tanh_layer.py"],
    deps = [
        "//pysyrenn/frontend:hard_tanh_layer",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "argmax_layer",
    size = "small",
    srcs = ["argmax_layer.py"],
    deps = [
        "//pysyrenn/frontend:argmax_layer",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "fullyconnected_layer",
    size = "small",
    srcs = ["fullyconnected_layer.py"],
    deps = [
        "//pysyrenn/frontend:fullyconnected_layer",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "normalize_layer",
    size = "small",
    srcs = ["normalize_layer.py"],
    deps = [
        "//pysyrenn/frontend:normalize_layer",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "averagepool_layer",
    size = "small",
    srcs = ["averagepool_layer.py"],
    deps = [
        "//pysyrenn/frontend:averagepool_layer",
        "//pysyrenn/frontend:strided_window_data",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "maxpool_layer",
    size = "small",
    srcs = ["maxpool_layer.py"],
    deps = [
        "//pysyrenn/frontend:maxpool_layer",
        "//pysyrenn/frontend:strided_window_data",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "conv2d_layer",
    size = "small",
    srcs = ["conv2d_layer.py"],
    deps = [
        "//pysyrenn/frontend:conv2d_layer",
        "//pysyrenn/frontend:strided_window_data",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "concat_layer",
    size = "small",
    srcs = ["concat_layer.py"],
    deps = [
        "//pysyrenn/frontend:averagepool_layer",
        "//pysyrenn/frontend:concat_layer",
        "//pysyrenn/frontend:conv2d_layer",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:normalize_layer",
        "//pysyrenn/frontend:strided_window_data",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "strided_window_data",
    size = "small",
    srcs = ["strided_window_data.py"],
    deps = [
        "//pysyrenn/frontend:strided_window_data",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "network",
    size = "medium",
    srcs = ["network.py"],
    data = ["@onnx_squeezenet//:all"],
    deps = [
        "//pysyrenn/frontend:argmax_layer",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:relu_layer",
        "//pysyrenn/frontend:transformer_client",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "layer",
    size = "small",
    srcs = ["layer.py"],
    deps = [
        "//pysyrenn/frontend:layer",
        "@bazel_python//:pytest_helper",
    ],
)

py_test(
    name = "transformer_client",
    size = "small",
    srcs = ["transformer_client.py"],
    deps = [
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:relu_layer",
        "//pysyrenn/frontend:transformer_client",
        "@bazel_python//:pytest_helper",
    ],
)
