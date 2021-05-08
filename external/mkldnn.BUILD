filegroup(
    name = "all",
    srcs = glob(["**"]),
)

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

# MKL-DNN for conv2d
cmake(
    name = "mkldnn",
    cache_entries = {
        # MKL-DNN's source throws set-but-not-used warnings with -Werror,
        # setting this turns those off
        "MKLDNN_PRODUCT_BUILD_MODE": "OFF",
        "WITH_TEST": "OFF",
        "WITH_EXAMPLE": "OFF",
    },
    lib_source = "all",
    out_shared_libs = [
        "libmkldnn.so.1.0",
        "libmkldnn.so.1",
        "libmkldnn.so",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)
