py_library(
    name = "helpers",
    srcs = ["__init__.py"],
    deps = [
        ":integrated_gradients",
        ":classify_lines",
        ":classify_planes",
        ":netpatch",
        ":masking_network",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "integrated_gradients",
    srcs = ["integrated_gradients.py"],
    deps = [
        "//:pip-packages",
    ],
    visibility = [":__subpackages__"],
)

py_library(
    name = "classify_lines",
    srcs = ["classify_lines.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:argmax_layer",
    ],
    visibility = [":__subpackages__"],
)

py_library(
    name = "classify_planes",
    srcs = ["classify_planes.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:argmax_layer",
    ],
    visibility = [":__subpackages__"],
)

py_library(
    name = "netpatch",
    srcs = ["netpatch.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:relu_layer",
        ":masking_network",
    ],
    visibility = [":__subpackages__"],
)

py_library(
    name = "masking_network",
    srcs = ["masking_network.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn/frontend:network",
        "//pysyrenn/frontend:fullyconnected_layer",
        "//pysyrenn/frontend:relu_layer",
        "//syrenn_proto:syrenn_py_proto",
    ],
    visibility = [":__subpackages__"],
)
