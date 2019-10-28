cc_library(
    name = "shared",
    srcs = [],
    hdrs = ["shared.h"],
    visibility = [":__subpackages__"],
    deps = [
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "segmented_line",
    srcs = [
        "segmented_line.cc",
    ],
    hdrs = ["segmented_line.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":shared",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "upolytope",
    srcs = ["upolytope.cc"],
    hdrs = ["upolytope.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        "@tbb",
    ],
)

cc_library(
    name = "transformer",
    srcs = ["transformer.cc"],
    hdrs = ["transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":upolytope",
    ],
)

cc_library(
    name = "pwl_transformer",
    srcs = ["pwl_transformer.cc"],
    hdrs = ["pwl_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
        "@tbb",
    ],
)

cc_library(
    name = "relu_transformer",
    srcs = ["relu_transformer.cc"],
    hdrs = ["relu_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
    ],
)

cc_library(
    name = "argmax_transformer",
    srcs = ["argmax_transformer.cc"],
    hdrs = ["argmax_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "hard_tanh_transformer",
    srcs = ["hard_tanh_transformer.cc"],
    hdrs = ["hard_tanh_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":pwl_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "affine_transformer",
    srcs = ["affine_transformer.cc"],
    hdrs = ["affine_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "fullyconnected_transformer",
    srcs = ["fullyconnected_transformer.cc"],
    hdrs = ["fullyconnected_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "normalize_transformer",
    srcs = ["normalize_transformer.cc"],
    hdrs = ["normalize_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
    ],
)

cc_library(
    name = "strided_window_data",
    srcs = ["strided_window_data.cc"],
    hdrs = ["strided_window_data.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":shared",
        "//syrenn_proto:syrenn_cc_grpc",
        "@mkldnn",
    ],
)

cc_library(
    name = "conv2d_transformer",
    srcs = ["conv2d_transformer.cc"],
    hdrs = ["conv2d_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
    ],
)

cc_library(
    name = "maxpool_transformer",
    srcs = ["maxpool_transformer.cc"],
    hdrs = ["maxpool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":segmented_line",
        ":shared",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
    ],
)

cc_library(
    name = "relu_maxpool_transformer",
    srcs = ["relu_maxpool_transformer.cc"],
    hdrs = ["relu_maxpool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":maxpool_transformer",
        ":relu_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
    ],
)

cc_library(
    name = "averagepool_transformer",
    srcs = ["averagepool_transformer.cc"],
    hdrs = ["averagepool_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":affine_transformer",
        ":segmented_line",
        ":shared",
        ":strided_window_data",
        ":transformer",
        "//syrenn_proto:syrenn_cc_grpc",
        "@eigen",
        "@mkldnn",
    ],
)

cc_library(
    name = "concat_transformer",
    srcs = ["concat_transformer.cc"],
    hdrs = ["concat_transformer.h"],
    visibility = [":__subpackages__"],
    deps = [
        ":conv2d_transformer",
        ":segmented_line",
        ":shared",
        ":transformer",
        "@eigen",
        "@mkldnn",
    ],
)

cc_binary(
    name = "server",
    srcs = ["server.cc"],
    visibility = [":__subpackages__"],
    deps = [
        ":argmax_transformer",
        ":averagepool_transformer",
        ":concat_transformer",
        ":conv2d_transformer",
        ":fullyconnected_transformer",
        ":hard_tanh_transformer",
        ":maxpool_transformer",
        ":normalize_transformer",
        ":relu_maxpool_transformer",
        ":relu_transformer",
        ":segmented_line",
        ":shared",
        ":upolytope",
        "//syrenn_proto:syrenn_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
    ],
)
