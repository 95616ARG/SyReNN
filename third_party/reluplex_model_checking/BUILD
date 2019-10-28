genrule(
    name = "build_reluplex_docker",
    srcs = [
        "Dockerfile",
        "bmc.cpp",
        "Makefile",
    ],
    outs = ["reluplex_bmc.tgz"],
    cmd = """
    cp $(location Dockerfile) Dockerfile
    cp $(location bmc.cpp) bmc.cpp
    cp $(location Makefile) Makefile
    docker build --force-rm -t reluplex_bmc .
    docker save -o reluplex_bmc.tgz reluplex_bmc
    cp reluplex_bmc.tgz $@
    """,
)

genrule(
    name = "export_specs",
    srcs = [
        "export_specs.py",
        "//models:vrl_models",
    ],
    outs = ["specs.tgz"],
    cmd = """
    SCRIPT=$(location export_specs.py)
    PYTHON=$(location //:pywrapper)
    export PYTHONPATH=$(location //:pip_packages)

    $$PYTHON $$SCRIPT
    tar -czvf specs.tgz pendulum_continuous satelite quadcopter
    cp specs.tgz $@
    """,
    tools = [
        "//:pip_packages",
        "//:pywrapper",
        "//experiments:vrl_model",
        "@python_3//:installdir",
    ],
    visibility = ["//:__subpackages__"],
)

sh_binary(
    name = "experiment",
    srcs = ["experiment.sh"],
    data = [
        ":reluplex_bmc.tgz",
        ":specs.tgz",
    ],
)
