filegroup(
    name = "vrl_models",
    srcs = glob([
        "vrl/eran/*.eran",
    ]),
    visibility = ["//:__subpackages__"],
)

filegroup(
    name = "models",
    srcs = [
        ":acas_models",
        ":vrl_models",
        "@cifar10_relu_4_100_model//file",
        "@cifar10_relu_9_200_model//file",
        "@cifar10_relu_convbig_diffai_model//file",
        "@cifar10_relu_convmedium_model//file",
        "@cifar10_relu_convsmall_diffai_model//file",
        "@cifar10_relu_convsmall_model//file",
        "@cifar10_relu_convsmall_pgd_model//file",
        "@mnist_relu_3_100_model//file",
        "@mnist_relu_4_1024_model//file",
        "@mnist_relu_convmedium_model//file",
        "@mnist_relu_convsmall_diffai_model//file",
        "@mnist_relu_convsmall_model//file",
        "@mnist_relu_convsmall_pgd_model//file",
    ],
    visibility = ["//:__subpackages__"],
)

filegroup(
    name = "datasets",
    srcs = [
        "@cifar10_test_data//file",
        "@mnist_test_data//file",
    ],
    visibility = ["//:__subpackages__"],
)

# Downloads ACAS Xu models and translates them into the ERAN model format.
genrule(
    name = "translate-acas-models",
    srcs = [
        "@reluplex//:all",
        "translate_acas_model.py",
    ],
    outs = ["acas_models"],
    cmd = """
    PYTHON_VENV=$(location //:bazel_python_venv)
    pushd $$PYTHON_VENV/..
    source bazel_python_venv_installed/bin/activate
    popd

    cp -L external/reluplex/nnet/*.nnet .
    all_files=$$(ls | grep .nnet)

    for file in $$all_files
    do
        python3 $(location translate_acas_model.py) $$file
    done

    mkdir acas_models
    mv *.eran acas_models
    mv acas_models $@
    """,
    tools = [
        "//:bazel_python_venv",
    ],
    visibility = ["//:__subpackages__"],
)
