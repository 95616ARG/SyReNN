filegroup(
    name = "all",
    srcs = glob(["**"]),
)

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "eigen",
    # These options help CMake to find prebuilt OpenBLAS, which will be copied into
    # $EXT_BUILD_DEPS/openblas by the cmake_external script
    cache_entries = {
        "BLAS_VENDOR": "OpenBLAS",
        "BLAS_LIBRARIES": "$EXT_BUILD_DEPS/openblas/lib/libopenblas.a",
    },
    lib_source = "all",
    out_headers_only = True,
    visibility = ["//visibility:public"],
    # Dependency on other cmake_external rule; can also depend on cc_import, cc_library rules
    deps = ["@openblas"],
)
