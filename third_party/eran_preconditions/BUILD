genrule(
    name = "build_eran_docker",
    srcs = [
        "Dockerfile",
        "install.sh",
    ],
    outs = ["eran_image.tgz"],
    cmd = """
    cp $(location Dockerfile) Dockerfile
    cp $(location install.sh) install.sh
    docker build --force-rm -t eran_image .
    docker save -o eran_image.tgz eran_image
    cp eran_image.tgz $@
    """,
    visibility = [
        "//third_party/eran_bmc:__pkg__",
    ],
)

sh_binary(
    name = "experiment",
    srcs = ["experiment.sh"],
    data = [
        "experiment.py",
        ":eran_image.tgz",
        "//:plane.png",
        "//experiments:polar_image_only",
        "//models:acas_models",
    ],
)
