genrule(
    name = "build_eran_docker",
    srcs = [
        "Dockerfile",
        "install.sh",
        "//:plane.png",
    ],
    outs = ["eran_preconditions.tgz"],
    cmd = """
    cp $(location //:plane.png) plane.png
    cp $(location Dockerfile) Dockerfile
    cp $(location install.sh) install.sh
    docker build --force-rm -t eran_preconditions .
    docker save -o eran_preconditions.tgz eran_preconditions
    cp eran_preconditions.tgz $@
    """,
)

sh_binary(
    name = "experiment",
    srcs = ["experiment.sh"],
    data = [
        "experiment.py",
        ":eran_preconditions.tgz",
        "//experiments:polar_image_only",
        "//models:acas_models",
    ],
)
