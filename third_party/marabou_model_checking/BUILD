genrule(
    name = "build_marabou_docker",
    srcs = [
        "Dockerfile",
        "bmc.cpp",
        "Makefile",
    ],
    outs = ["marabou_bmc.tgz"],
    cmd = """
    cp $(location Dockerfile) Dockerfile
    cp $(location bmc.cpp) bmc.cpp
    cp $(location Makefile) Makefile
    docker build --force-rm -t marabou_bmc .
    docker save -o marabou_bmc.tgz marabou_bmc
    cp marabou_bmc.tgz $@
    """,
)

sh_binary(
    name = "experiment",
    srcs = ["experiment.sh"],
    data = [
        ":marabou_bmc.tgz",
        "//third_party/reluplex_model_checking:specs.tgz",
    ],
)
