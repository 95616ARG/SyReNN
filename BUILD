load("@bazel_python//:bazel_python.bzl", "bazel_python_interpreter")

bazel_python_interpreter(
    name = "bazel_python_venv",
    python_version = "3.7.4",
    requirements_file = "requirements.txt",
    run_after_pip = """
        pip3 install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    """,
    visibility = ["//:__subpackages__"],
)

# Make the thicker-bordered plane SVG.
genrule(
    name = "thicker-plane",
    srcs = ["@plane_svg//file"],
    outs = ["plane.png"],
    cmd = """
    PLANESVG=$(location @plane_svg//file)

    PYTHON_VENV=$(location //:bazel_python_venv)
    pushd $$PYTHON_VENV/..
    source bazel_python_venv_installed/bin/activate
    popd

    sed -i -e \
        's/id="path5724" /id="path5724" stroke="white" fill="black" stroke-width="10" /' \
        $$PLANESVG

    python3 -m cairosvg $$PLANESVG -o plane.png --output-width 4965
    cp plane.png $@
    """,
    tools = [
        "//:bazel_python_venv",
    ],
    visibility = ["//:__subpackages__"],
)

# For generating the coverage report.
sh_binary(
    name = "coverage_report",
    srcs = ["coverage_report.sh"],
    deps = [":_dummy_coverage_report"],
)

# This is only to get bazel_python_venv as a data dependency for
# coverage_report above. For some reason, this doesn't work if we directly put
# it on the sh_binary. This is a known issue:
# https://github.com/bazelbuild/bazel/issues/1147#issuecomment-428698802
sh_library(
    name = "_dummy_coverage_report",
    srcs = ["coverage_report.sh"],
    data = ["//:bazel_python_venv"],
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
    PYTHON_VENV=$(location //:bazel_python_venv)
    pushd $$PYTHON_VENV/..
    source bazel_python_venv_installed/bin/activate
    popd

    mkdir -p syrenn_proto
    cp -Lr $(locations //syrenn_proto:syrenn_py_grpc) syrenn_proto
    cp -Lr $(locations //syrenn_proto:syrenn_py_proto) syrenn_proto
    touch syrenn_proto/__init__.py
    cp pip_info/* .
    python3 setup.py sdist bdist_wheel

    cp -r dist $@
    """,
    tools = [
        "//:bazel_python_venv",
        "//syrenn_proto:syrenn_py_grpc",
        "//syrenn_proto:syrenn_py_proto",
    ],
)
