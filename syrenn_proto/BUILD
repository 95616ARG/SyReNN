load("@build_stack_rules_proto//cpp:cpp_grpc_library.bzl", "cpp_grpc_library")
load("@build_stack_rules_proto//python:python_grpc_library.bzl", "python_grpc_library")

proto_library(
    name = "syrenn_proto",
    srcs = ["syrenn.proto"],
    visibility = ["//visibility:public"],
)

cpp_grpc_library(
    name = "syrenn_cc_proto",
    deps = [":syrenn_proto"],
    has_services = True,
    visibility = ["//visibility:public"],
)

python_grpc_library(
    name = "syrenn_py_proto",
    deps = [":syrenn_proto"],
    has_services = True,
    visibility = ["//visibility:public"],
)
