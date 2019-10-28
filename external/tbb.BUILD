# TBB, adapted from https://github.com/tensorflow/tensorflow/blob/master/third_party/ngraph/tbb.BUILD
genrule(
    name = "build_tbb",
    srcs = glob(["**/**"]) + [
        "@local_config_cc//:toolchain",
    ],
    outs = [
        "libtbb.a",
        "libtbbmalloc.a",
    ],
    cmd = """
         DEST_DIR=$$PWD/$(@D)
         export CXX=gcc
         #TBB's build needs some help to figure out what compiler it's using
         export COMPILER_OPT="compiler=gcc"
         # Workaround for TBB bug
         # See https://github.com/01org/tbb/issues/59
         # export CXXFLAGS="-flifetime-dse=1"
         cd external/tbb
         # uses extra_inc=big_iron.inc to specify that static libraries are
         # built. See https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/297792
         make tbb_build_prefix="build" \
              extra_inc=big_iron.inc \
              $$COMPILER_OPT; \
         # echo cp build/build_{release,debug}/*.a $$DEST_DIR
         cp build/build_{release,debug}/*.a $$DEST_DIR
    """,
)

cc_library(
    name = "tbb",
    srcs = [
        "libtbb.a",
        ":build_tbb",
    ],
    hdrs = glob([
        "include/serial/**",
        "include/tbb/**/**",
    ]),
    includes = ["include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
