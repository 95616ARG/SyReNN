cc_test(
    name = "relu_transformer",
    srcs = ["relu_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "@eigen//:eigen",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:relu_transformer",
    ],
)

cc_test(
    name = "argmax_transformer",
    srcs = ["argmax_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "@eigen//:eigen",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:argmax_transformer",
    ],
)

cc_test(
    name = "affine_transformer",
    srcs = ["affine_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "@eigen//:eigen",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:affine_transformer",
    ],
)

cc_test(
    name = "fullyconnected_transformer",
    srcs = ["fullyconnected_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "@eigen//:eigen",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:fullyconnected_transformer",
    ],
)

cc_test(
    name = "normalize_transformer",
    srcs = ["normalize_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "@eigen//:eigen",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:normalize_transformer",
    ],
)

cc_test(
    name = "hard_tanh_transformer",
    srcs = ["hard_tanh_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:hard_tanh_transformer",
        "//syrenn_proto:syrenn_cc_proto",
    ],
)

cc_test(
    name = "strided_window_data",
    srcs = ["strided_window_data.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:strided_window_data",
        "//syrenn_proto:syrenn_cc_proto",
    ],
)

cc_test(
    name = "averagepool_transformer",
    srcs = ["averagepool_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "@eigen//:eigen",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:averagepool_transformer",
        "//syrenn_server:strided_window_data",
    ],
)

cc_test(
    name = "maxpool_transformer",
    srcs = ["maxpool_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:segmented_line",
        "//syrenn_server:maxpool_transformer",
        "//syrenn_server:strided_window_data",
    ],
)

cc_test(
    name = "relu_maxpool_transformer",
    srcs = ["relu_maxpool_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:segmented_line",
        "//syrenn_server:relu_maxpool_transformer",
        "//syrenn_server:strided_window_data",
    ],
)

cc_test(
    name = "conv2d_transformer",
    srcs = ["conv2d_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:conv2d_transformer",
        "//syrenn_server:strided_window_data",
    ],
)

cc_test(
    name = "pwl_transformer",
    srcs = ["pwl_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:pwl_transformer",
    ],
)

cc_test(
    name = "concat_transformer",
    srcs = ["concat_transformer.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:segmented_line",
        "//syrenn_server:upolytope",
        "//syrenn_server:fullyconnected_transformer",
        "//syrenn_server:relu_transformer",
        "//syrenn_server:conv2d_transformer",
        "//syrenn_server:concat_transformer",
    ],
)

cc_test(
    name = "segmented_line",
    srcs = ["segmented_line.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:segmented_line",
    ],
)

cc_test(
    name = "upolytope",
    srcs = ["upolytope.cc"],
    copts = ["-Iexternal/gtest/include"],
    size = "small",
    deps = [
        "@gtest//:main",
        "//syrenn_server:upolytope",
    ],
)
