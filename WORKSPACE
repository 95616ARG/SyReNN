workspace(name = "SyReNN")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "bazel_python",
    commit = "cb4f346641afd090dc088c01df46dc403e600773",
    remote = "https://github.com/95616ARG/bazel_python.git",
)

load("@bazel_python//:bazel_python.bzl", "bazel_python")

bazel_python()

# EIGEN SUPPORT
# See the README in: https://github.com/bazelbuild/rules_foreign_cc
# Group the sources of the library so that CMake rule have access to it
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# Rule repository
git_repository(
    name = "rules_foreign_cc",
    commit = "b8b88cd2d16035aa1639434eb808f4d67a34d5ae",
    remote = "https://github.com/bazelbuild/rules_foreign_cc.git",
    shallow_since = "1620401997 -0700",
)
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()

# OpenBLAS, Eigen, and MKLDNN source code repositories
http_archive(
    name = "openblas",
    build_file = "openblas.BUILD",
    sha256 = "488294c6176bd0318b2453d0cff203a31f76a336a06c48f8ae7a23e46bf4d5df",
    strip_prefix = "OpenBLAS-744779d3351aa96afa5188d75eb76ac5ec197d13",
    urls = ["https://github.com/xianyi/OpenBLAS/archive/744779d3351aa96afa5188d75eb76ac5ec197d13.tar.gz"],
)

http_archive(
    name = "eigen",
    build_file = "eigen.BUILD",
    sha256 = "992855522dfdd0dea74d903dcd082cdb01c1ae72be5145e2fe646a0892989e43",
    strip_prefix = "eigen-git-mirror-3.3.5",
    urls = ["https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"],
)

http_archive(
    name = "mkldnn",
    build_file = "mkldnn.BUILD",
    sha256 = "8fee2324267811204c1f877a1dea70b23ab3d5f4c3ea0198d81f0921aa70d76e",
    strip_prefix = "oneDNN-1.0.1",
    urls = ["https://github.com/oneapi-src/oneDNN/archive/v1.0.1.tar.gz"],
)

http_archive(
    name = "tbb",
    build_file = "tbb.BUILD",
    sha256 = "6b540118cbc79f9cbc06a35033c18156c21b84ab7b6cf56d773b168ad2b68566",
    strip_prefix = "oneTBB-2019_U8",
    urls = ["https://github.com/oneapi-src/oneTBB/archive/2019_U8.tar.gz"],
)

# GOOGLETEST
# https://docs.bazel.build/versions/master/cpp-use-cases.html#writing-and-running-c-tests
http_archive(
    name = "gtest",
    build_file = "gtest.BUILD",
    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    strip_prefix = "googletest-release-1.7.0",
    url = "https://github.com/google/googletest/archive/release-1.7.0.zip",
)

##### BEGIN GRPC #####
git_repository(
    name = "com_github_grpc_grpc",
    commit = "8664c8334c05d322fbbdfb9e3b24601a23e9363c",
    shallow_since = "1619560885 -0700",
    remote = "https://github.com/grpc/grpc.git",
)
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
# This is annoying and bloated because it assumes we want, e.g., Go support.
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

##### END GRPC #####

# Plane image from Wikimedia Commons.
http_file(
    name = "plane_svg",
    downloaded_file_path = "plane.svg",
    sha256 = "f4bec6365e51afdba8fa254247d44e19f3beb81eadfe369ec0867ee9959d1a49",
    urls = ["https://upload.wikimedia.org/wikipedia/commons/9/95/Plane_icon_nose_down.svg"],
)

# From https://github.com/eth-sri/eran

# TODO(masotoud): Find a way to not have to do all of these explicitly. Perhaps
# download a zip of the ERAN repo and just pull out the models? This will
# require modifying the run_analysis, etc. code re: where it pulls models from.

# MODELS
http_file(
    name = "mnist_relu_3_100_model",
    downloaded_file_path = "model.eran",
    sha256 = "e4151dfced1783360ab8353c8fdedbfd76f712c2c56e4b14799b2f989217229f",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf"],
)

http_file(
    name = "mnist_relu_4_1024_model",
    downloaded_file_path = "model.eran",
    sha256 = "03b53b317dfd0a05a72a7566d80d7fc4756a2e664c93c93459b12cd7b886d810",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_4_1024.tf"],
)

http_file(
    name = "mnist_relu_convsmall_model",
    downloaded_file_path = "model.eran",
    sha256 = "87953b7f8412091e9eebaae7c56e5432b4a1559b6290676c559f4a7f9825cfc7",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__Point.pyt"],
)

http_file(
    name = "mnist_relu_convsmall_diffai_model",
    downloaded_file_path = "model.eran",
    sha256 = "76371053d6faac369858769a25b845bfd177bd88a693b3c49e4590ee3f2fe721",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__DiffAI.pyt"],
)

http_file(
    name = "mnist_relu_convsmall_pgd_model",
    downloaded_file_path = "model.eran",
    sha256 = "1cad08171eb1651e49aa8ae13855c3c3dfa125dac32f7187253139d267e43f7e",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__PGDK.pyt"],
)

http_file(
    name = "mnist_relu_convmedium_model",
    downloaded_file_path = "model.eran",
    sha256 = "88c414a0f69a3469731c45bc14d9583cb8189fbbbab003ca70d818a041a31103",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__Point.pyt"],
)

http_file(
    name = "cifar10_relu_4_100_model",
    downloaded_file_path = "model.eran",
    sha256 = "9f72f9f798109264c02d4d8ad8d7d2e6fcfd8d474fb7e0f9b482741279a4d780",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_4_100.tf"],
)

http_file(
    name = "cifar10_relu_9_200_model",
    downloaded_file_path = "model.eran",
    sha256 = "46964f58e6fce152a51e0fe69620c91527497eb0152ea95d926f167b88c67d01",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_9_200.tf"],
)

http_file(
    name = "cifar10_relu_convsmall_model",
    downloaded_file_path = "model.eran",
    sha256 = "e486e5adc14ec474beeddeabf71ceb6aa50aa4e789ab23d4ebb9df71e58eadaa",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__Point.pyt"],
)

http_file(
    name = "cifar10_relu_convsmall_diffai_model",
    downloaded_file_path = "model.eran",
    sha256 = "e9dceef1ed18b488ffbfa22649725d7fdb176fa97e06fd47101015f94d62313d",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__DiffAI.pyt"],
)

http_file(
    name = "cifar10_relu_convsmall_pgd_model",
    downloaded_file_path = "model.eran",
    sha256 = "83e7b7e426a35da490924947eb7792634bced5f9723cf24d0c238624122b6fa8",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__PGDK.pyt"],
)

http_file(
    name = "cifar10_relu_convmedium_model",
    downloaded_file_path = "model.eran",
    sha256 = "136df5d070cef55268316965a2aa1bc68c0b61da4685a331619ee3d8a9e97e50",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__Point.pyt"],
)

http_file(
    name = "cifar10_relu_convbig_diffai_model",
    downloaded_file_path = "model.eran",
    sha256 = "f2265c24149ba0ba04480a315d54e2301870149f90da039418b31332d03781a0",
    urls = ["https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convBigRELU__DiffAI.pyt"],
)

# ACAS MODELS
http_archive(
    name = "reluplex",
    build_file_content = all_content,
    sha256 = "d058b1a7873416f2581a975e0099f5145888faf711b6f7e9b8f1a9b61d75f203",
    strip_prefix = "ReluplexCav2017-c6fcc3308c3edcf1c155e2b9625ac924d06e099b",
    urls = ["https://github.com/guykatzz/ReluplexCav2017/archive/c6fcc3308c3edcf1c155e2b9625ac924d06e099b.zip"],
)

# ONNX TEST MODEL
http_archive(
    name = "onnx_squeezenet",
    build_file_content = all_content,
    sha256 = "aff6280d73c0b826f088f7289e4495f01f6e84ce75507279e1b2a01590427723",
    strip_prefix = "squeezenet1.1",
    urls = ["https://s3.amazonaws.com/onnx-model-zoo/squeezenet/squeezenet1.1/squeezenet1.1.tar.gz"],
)

# DATASETS
http_file(
    name = "mnist_test_data",
    downloaded_file_path = "data.csv",
    sha256 = "91059c3f780826c6474841e3ec7ca6aad4c89878753a885704c868306573dd29",
    urls = ["https://raw.githubusercontent.com/eth-sri/eran/16efb21c401c6912327d6fb70b237ae7a8a1ccd9/data/mnist_test.csv"],
)

http_file(
    name = "cifar10_test_data",
    downloaded_file_path = "data.csv",
    sha256 = "4cef6cdf54a6b78c7cae5ac7d20dc1d6eda1c9d8a04c6b08b5ecba6029a6f043",
    urls = ["https://raw.githubusercontent.com/eth-sri/eran/16efb21c401c6912327d6fb70b237ae7a8a1ccd9/data/cifar10_test.csv"],
)

http_archive(
    name = "imagenette",
    build_file_content = all_content,
    sha256 = "0e3ac85d7f3fd0a63d5e27d85c68511be9ac4dae260345b6f7434a0d4fa24b3f",
    strip_prefix = "imagenette-320",
    urls = ["https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz"],
)
