py_library(
    name = "helpers",
    srcs = ["__init__.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":classify_lines",
        ":classify_planes",
        ":integrated_gradients",
        ":masking_network",
        ":netpatch",
    ],
)

py_library(
    name = "integrated_gradients",
    srcs = ["integrated_gradients.py"],
    visibility = [":__subpackages__"],
)

py_library(
    name = "classify_lines",
    srcs = ["classify_lines.py"],
    visibility = [":__subpackages__"],
    deps = [
        "//pysyrenn/frontend:argmax_layer",
        "//pysyrenn/frontend:network",
    ],
)

py_library(
    name = "classify_planes",
    srcs = ["classify_planes.py"],
    visibility = [":__subpackages__"],
    deps = [
        "//pysyrenn/frontend:argmax_layer",
        "//pysyrenn/frontend:network",
    ],
)

py_library(
    name = "netpatch",
    srcs = ["netpatch.py"],
    visibility = [":__subpackages__"],
)

py_library(
    name = "masking_network",
    srcs = ["masking_network.py"],
    visibility = [":__subpackages__"],
)
