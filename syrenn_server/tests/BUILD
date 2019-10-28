cc_test(
    name = "relu_transformer",
    size = "small",
    srcs = ["relu_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:relu_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_test(
    name = "argmax_transformer",
    size = "small",
    srcs = ["argmax_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:argmax_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_test(
    name = "affine_transformer",
    size = "small",
    srcs = ["affine_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:affine_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_test(
    name = "fullyconnected_transformer",
    size = "small",
    srcs = ["fullyconnected_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:fullyconnected_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_test(
    name = "normalize_transformer",
    size = "small",
    srcs = ["normalize_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:normalize_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_test(
    name = "hard_tanh_transformer",
    size = "small",
    srcs = ["hard_tanh_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_proto:syrenn_cc_grpc",
        "//syrenn_server:hard_tanh_transformer",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "strided_window_data",
    size = "small",
    srcs = ["strided_window_data.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_proto:syrenn_cc_grpc",
        "//syrenn_server:strided_window_data",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "averagepool_transformer",
    size = "small",
    srcs = ["averagepool_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:averagepool_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:strided_window_data",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@eigen",
        "@gtest//:main",
    ],
)

cc_test(
    name = "maxpool_transformer",
    size = "small",
    srcs = ["maxpool_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:maxpool_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:strided_window_data",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "relu_maxpool_transformer",
    size = "small",
    srcs = ["relu_maxpool_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:relu_maxpool_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:strided_window_data",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "conv2d_transformer",
    size = "small",
    srcs = ["conv2d_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:conv2d_transformer",
        "//syrenn_server:strided_window_data",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "pwl_transformer",
    size = "small",
    srcs = ["pwl_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:pwl_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "concat_transformer",
    size = "small",
    srcs = ["concat_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:concat_transformer",
        "//syrenn_server:conv2d_transformer",
        "//syrenn_server:fullyconnected_transformer",
        "//syrenn_server:relu_transformer",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "segmented_line",
    size = "small",
    srcs = ["segmented_line.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:segmented_line",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)

cc_test(
    name = "upolytope",
    size = "small",
    srcs = ["upolytope.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "//syrenn_server:upolytope",
        "@com_github_grpc_grpc//:grpc++",
        "@gtest//:main",
    ],
)
