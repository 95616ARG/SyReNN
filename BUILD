sh_library(
    name = "pywrapper",
    srcs = ["pywrapper.sh"],
    visibility = ["//:__subpackages__"],
)

# https://stackoverflow.com/questions/47036855
py_runtime(
    name = "python-3",
    files = ["@python_3//:installdir"],
    interpreter = "pywrapper.sh",
    python_version = "PY3",
)

# https://github.com/bazelbuild/rules_python/blob/master/proposals/2019-02-12-design-for-a-python-toolchain.md
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

constraint_value(
    name = "python3_constraint",
    constraint_setting = "@bazel_tools//tools/python:py3_interpreter_path",
)

platform(
    name = "python3_platform",
    constraint_values = [
        ":python3_constraint",
    ],
)

py_runtime_pair(
    name = "python3_runtime_pair",
    py3_runtime = ":python-3",
)

toolchain(
    name = "python3-toolchain",
    # Since the Python interpreter is invoked at runtime on the target
    # platform, there's no need to specify execution platform constraints here.
    target_compatible_with = [
        # Make sure this toolchain is only selected for a target platform that
        # advertises that it has interpreters available under /usr/weirdpath.
        # ":python3_constraint",
    ],
    toolchain = "//:python3_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

# This seems to be the way Google is doing it now:
# https://github.com/bazelbuild/rules_python/issues/119
# See https://github.com/pypa/pip/issues/3826 for why we need the --system flag
# on the pip call. For newer versions of pip, you may have to add
# --no-build-isolation.
genrule(
    name = "install-pip-packages",
    srcs = ["requirements.txt"],
    outs = ["pip_packages"],
    cmd = """
    PYTHON=$(location pywrapper.sh)
    PIP="$$PYTHON -m pip"

    DUMMY_HOME=/tmp/$$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 8)
    rm -rf $$DUMMY_HOME
    export HOME=$$DUMMY_HOME

    PIP_INSTALL="$$PIP \
        install --no-cache-dir --disable-pip-version-check \
        --target=$@"

    # Install the correct version of Torch
    mkdir -p $$DUMMY_HOME
    $$PIP_INSTALL torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

    # Install the other requirements.
    mkdir -p $$DUMMY_HOME
    $$PIP_INSTALL -r requirements.txt

    # The custom typing package installed as a dependency doesn't seem to work
    # well.
    rm -rf $@/typing-*
    rm -rf $@/typing.py

    rm -rf $$DUMMY_HOME
    """,
    tools = [
        ":pywrapper.sh",
        "@python_3//:installdir",
    ],
    visibility = ["//:__subpackages__"],
)

py_library(
    name = "pip-packages",
    srcs = ["._dummy_.py"],
    data = [":install-pip-packages"],
    imports = ["pip_packages"],
    visibility = ["//:__subpackages__"],
)

# Make the thicker-bordered plane SVG.
genrule(
    name = "thicker-plane",
    srcs = [
        "@plane_svg//file",
        "pywrapper.sh",
    ],
    outs = ["plane.png"],
    cmd = """
    PLANESVG=$(location @plane_svg//file)
    PYTHON=$(location pywrapper.sh)
    export PYTHONPATH=$(location //:pip_packages)

    sed -i -e \
        's/id="path5724" /id="path5724" stroke="white" fill="black" stroke-width="10" /' \
        $$PLANESVG

    $$PYTHON -m cairosvg $$PLANESVG -o plane.png --output-width 4965
    cp plane.png $@
    """,
    tools = [
        "//:pip_packages",
        "@python_3//:installdir",
    ],
    visibility = ["//:__subpackages__"],
)

# For generating the coverage report.
sh_binary(
    name = "coverage_report",
    srcs = ["coverage_report.sh"],
    data = [
        "//:pip_packages",
        "//:pywrapper",
        "@python_3//:installdir",
    ],
)

# For wheel-ifying the Python code.
# Thanks!
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
genrule(
    name = "wheel",
    srcs = [
        "pysyrenn",
        "requirements.txt",
        "LICENSE",
        "pip_info/__metadata__.py",
        "pip_info/README.md",
        "pip_info/setup.cfg",
        "pip_info/setup.py",
    ],
    outs = ["pysyrenn.dist"],
    cmd = """
    PYTHON=$(location pywrapper)
    export PYTHONPATH=$$PWD:$(location //:pip_packages)

    mkdir -p syrenn_proto
    cp -Lr $(locations //syrenn_proto:syrenn_py_grpc) syrenn_proto
    cp -Lr $(locations //syrenn_proto:syrenn_py_proto) syrenn_proto
    touch syrenn_proto/__init__.py
    cp pip_info/* .
    $$PYTHON setup.py sdist bdist_wheel

    cp -r dist $@
    """,
    tools = [
        "pywrapper",
        "//:pip_packages",
        "//syrenn_proto:syrenn_py_grpc",
        "//syrenn_proto:syrenn_py_proto",
        "@python_3//:installdir",
    ],
)
