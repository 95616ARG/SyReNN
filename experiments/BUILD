py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    data = [
        "//models:models",
        "//models:datasets",
    ],
    deps = [
        "//:pip-packages",
        "//syrenn_proto:syrenn_py_proto",
        "//pysyrenn:pysyrenn",
    ],
)

py_library(
    name = "polar_image",
    srcs = ["polar_image.py"],
    deps = ["//:pip-packages"],
)

filegroup(
    name = "polar_image_only",
    srcs = [
        "polar_image.py",
    ],
    visibility = ["//:__subpackages__"],
)

py_binary(
    name = "integrated_gradients",
    srcs = ["integrated_gradients.py"],
    deps = [
        "//:pip-packages",
        ":experiment",
        "//pysyrenn:pysyrenn",
    ],
)

py_binary(
    name = "acas_lines",
    srcs = ["acas_lines.py"],
    deps = [
        "//:pip-packages",
        ":experiment",
        "//pysyrenn:pysyrenn",
    ],
    data = [
        "//:plane.png",
    ],
)

py_binary(
    name = "linearity_hypothesis",
    srcs = ["linearity_hypothesis.py"],
    deps = [
        "//:pip-packages",
        ":experiment",
        "//pysyrenn:pysyrenn",
    ],
)

py_binary(
    name = "acas_planes",
    srcs = ["acas_planes.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn:pysyrenn",
        ":acas_lines",
        ":polar_image",
    ],
    data = [
        "//:plane.png",
    ],
)

py_library(
    name = "vrl_model",
    srcs = ["vrl_model.py"],
    deps = ["//:pip-packages"],
    visibility = ["//:__subpackages__"],
)

py_binary(
    name = "model_checking",
    srcs = ["model_checking.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn:pysyrenn",
        ":experiment",
        ":vrl_model",
    ],
    visibility = ["//:__subpackages__"],
)

py_binary(
    name = "netpatch",
    srcs = ["netpatch.py"],
    deps = [
        "//:pip-packages",
        "//pysyrenn:pysyrenn",
        ":acas_planes",
        ":polar_image",
    ],
)

py_binary(
    name = "toy_examples",
    srcs = ["toy_examples.py"],
    deps = [
        "//:pip-packages",
        ":acas_planes",
        "//pysyrenn:pysyrenn",
    ],
)
