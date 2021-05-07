py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    data = [
        "//models",
        "//models:datasets",
    ],
    deps = [
        "//pysyrenn",
        "//syrenn_proto:syrenn_py_grpc",
    ],
)

py_library(
    name = "polar_image",
    srcs = ["polar_image.py"],
)

filegroup(
    name = "polar_image_only",
    srcs = [
        "polar_image.py",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "integral_approximations",
    srcs = ["integral_approximations.py"],
)

py_binary(
    name = "integrated_gradients",
    srcs = ["integrated_gradients.py"],
    deps = [
        ":experiment",
        ":integral_approximations",
        "//pysyrenn",
    ],
)

py_binary(
    name = "acas_lines",
    srcs = ["acas_lines.py"],
    data = [
        "//:plane.png",
    ],
    deps = [
        ":experiment",
        "//pysyrenn",
    ],
)

py_binary(
    name = "linearity_hypothesis",
    srcs = ["linearity_hypothesis.py"],
    deps = [
        ":experiment",
        "//pysyrenn",
    ],
)

py_binary(
    name = "acas_planes",
    srcs = ["acas_planes.py"],
    data = [
        "//:plane.png",
    ],
    deps = [
        ":acas_lines",
        ":polar_image",
        "//pysyrenn",
    ],
)

py_library(
    name = "vrl_model",
    srcs = ["vrl_model.py"],
    visibility = ["//:__subpackages__"],
)

py_binary(
    name = "model_checking",
    srcs = ["model_checking.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":experiment",
        ":vrl_model",
        "//pysyrenn",
    ],
)

py_binary(
    name = "toy_examples",
    srcs = ["toy_examples.py"],
    deps = [
        ":acas_planes",
        "//pysyrenn",
    ],
)
