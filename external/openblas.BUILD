filegroup(
    name = "all",
    srcs = glob(["**"]),
)

load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

cmake_external(
    name = "openblas",
    # Values to be passed as -Dkey=value on the CMake command line;
    # here are serving to provide some CMake script configuration options
    cache_entries = {
        "NOFORTRAN": "on",
        "BUILD_WITHOUT_LAPACK": "no",
        "USE_OPENMP": "1",
    },
    lib_source = "all",
    linkopts = ["-lpthread"],
    make_commands = [
        "make -j4",
        "make install",
    ],

    # We are selecting the resulting static library to be passed in C/C++ provider
    # as the result of the build;
    # However, the cmake_external dependants could use other artefacts provided by the build,
    # according to their CMake script
    static_libraries = ["libopenblas.a"],
    visibility = ["//visibility:public"],
    alwayslink = True,
)
