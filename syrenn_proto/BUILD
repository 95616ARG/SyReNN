package(
    default_visibility = ["//visibility:public"],
)

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_grpc_library", "py_proto_library")

proto_library(
    name = "syrenn_proto",
    srcs = ["syrenn.proto"],
)

cc_proto_library(
    name = "syrenn_cc_proto",
    deps = [":syrenn_proto"],
)

cc_grpc_library(
    name = "syrenn_cc_grpc",
    srcs = [":syrenn_proto"],
    grpc_only = True,
    deps = [":syrenn_cc_proto"],
)

py_proto_library(
    name = "syrenn_py_proto",
    deps = [":syrenn_proto"],
)

py_grpc_library(
    name = "syrenn_py_grpc",
    srcs = [":syrenn_proto"],
    deps = [":syrenn_py_proto"],
)
