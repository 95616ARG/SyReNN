filegroup(
    name = "all",
    srcs = glob(["**"]),
)

load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

# MKL-DNN for conv2d
cmake_external(
    name = "mkldnn",
    cache_entries = {
        # MKL-DNN's source throws set-but-not-used warnings with -Werror,
        # setting this turns those off
        "MKLDNN_PRODUCT_BUILD_MODE" : "OFF",
        "WITH_TEST" : "OFF",
        "WITH_EXAMPLE" : "OFF",
    },
    alwayslink = True,
    lib_source = "all",
    make_commands = ["make -j8", "make install", "ls"],
    shared_libraries = ["libmkldnn.so.1"],
    visibility = ["//visibility:public"],
)
