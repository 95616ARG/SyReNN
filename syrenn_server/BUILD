cc_library(
    name = "shared",
    srcs = [],
    hdrs = ["shared.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        "@eigen//:eigen",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "segmented_line",
    srcs = [
        "segmented_line.cc"
    ],
    hdrs = ["segmented_line.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        "@eigen//:eigen",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "upolytope",
    srcs = ["upolytope.cc"],
    hdrs = ["upolytope.h"],
    deps = [
        ":segmented_line",
        ":shared",
        "@tbb//:tbb",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "transformer",
    srcs = ["transformer.cc"],
    hdrs = ["transformer.h"],
    deps = [
        ":segmented_line",
        ":shared",
        ":upolytope",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "pwl_transformer",
    srcs = ["pwl_transformer.cc"],
    hdrs = ["pwl_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":shared",
        ":upolytope",
        ":transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        "@tbb//:tbb",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "relu_transformer",
    srcs = ["relu_transformer.cc"],
    hdrs = ["relu_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":upolytope",
        ":transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        ":pwl_transformer",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "argmax_transformer",
    srcs = ["argmax_transformer.cc"],
    hdrs = ["argmax_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":upolytope",
        ":transformer",
        "@eigen//:eigen",
        ":pwl_transformer",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "hard_tanh_transformer",
    srcs = ["hard_tanh_transformer.cc"],
    hdrs = ["hard_tanh_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":transformer",
        "@eigen//:eigen",
        ":pwl_transformer",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "affine_transformer",
    srcs = ["affine_transformer.cc"],
    hdrs = ["affine_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":upolytope",
        ":transformer",
        "@eigen//:eigen",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "fullyconnected_transformer",
    srcs = ["fullyconnected_transformer.cc"],
    hdrs = ["fullyconnected_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":upolytope",
        ":transformer",
        ":affine_transformer",
        "@eigen//:eigen",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "normalize_transformer",
    srcs = ["normalize_transformer.cc"],
    hdrs = ["normalize_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":transformer",
        ":affine_transformer",
        "@eigen//:eigen",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "strided_window_data",
    srcs = ["strided_window_data.cc"],
    hdrs = ["strided_window_data.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        "@mkldnn//:mkldnn",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "conv2d_transformer",
    srcs = ["conv2d_transformer.cc"],
    hdrs = ["conv2d_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":transformer",
        ":affine_transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        ":shared",
        ":strided_window_data",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "maxpool_transformer",
    srcs = ["maxpool_transformer.cc"],
    hdrs = ["maxpool_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        ":shared",
        ":strided_window_data",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "relu_maxpool_transformer",
    srcs = ["relu_maxpool_transformer.cc"],
    hdrs = ["relu_maxpool_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":transformer",
        ":maxpool_transformer",
        ":relu_transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "averagepool_transformer",
    srcs = ["averagepool_transformer.cc"],
    hdrs = ["averagepool_transformer.h"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":transformer",
        ":affine_transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        ":shared",
        ":strided_window_data",
    ],
    visibility = [":__subpackages__"],
)

cc_library(
    name = "concat_transformer",
    srcs = ["concat_transformer.cc"],
    hdrs = ["concat_transformer.h"],
    deps = [
        ":segmented_line",
        ":transformer",
        ":conv2d_transformer",
        "@eigen//:eigen",
        "@mkldnn//:mkldnn",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)

cc_binary(
    name = "server",
    srcs = ["server.cc"],
    deps = [
        "//syrenn_proto:syrenn_cc_proto",
        ":segmented_line",
        ":upolytope",
        ":relu_transformer",
        ":hard_tanh_transformer",
        ":fullyconnected_transformer",
        ":conv2d_transformer",
        ":normalize_transformer",
        ":maxpool_transformer",
        ":relu_maxpool_transformer",
        ":averagepool_transformer",
        ":concat_transformer",
        ":argmax_transformer",
        "@eigen//:eigen",
        ":shared",
    ],
    visibility = [":__subpackages__"],
)
