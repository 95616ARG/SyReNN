#include "syrenn_server/relu_transformer.h"
#include <memory>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "mkldnn.hpp"

std::unique_ptr<LayerTransformer> ReLUTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_relu_data()) {
    return nullptr;
  }
  return std::unique_ptr<ReLUTransformer>(new ReLUTransformer());
}

size_t ReLUTransformer::n_piece_faces(size_t dims) const {
  return dims;
}

double ReLUTransformer::CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                                      Eigen::Ref<const RMVectorXf> to,
                                      const size_t face) const {
  return -from(face) / (to(face) - from(face));
}

int ReLUTransformer::PointSign(Eigen::Ref<const RMVectorXf> point,
                               const size_t face) const {
  if (point(face) == 0) {
    return 0;
  }
  return point(face) > 0 ? +1 : -1;
}

void ReLUTransformer::Compute(RMMatrixXf *inout) const {
  // Modified from
  // https://github.com/intel/mkl-dnn/blob/mnt-v0/examples/simple_net.cpp
  // See conv2d_transformer.cc for more.

  mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
  mkldnn::stream cpu_stream(cpu_engine);

  // NOTE: MKL reads the dimension sizes in NCHW even though the layout we
  // store it in is NHWC
  mkldnn::memory::dims input_dimensions{static_cast<int>(inout->size())};

  // MKL memory references to the above buffers
  auto inout_memory =
      mkldnn::memory(
              {
                  { input_dimensions },
                  mkldnn::memory::data_type::f32,
                  mkldnn::memory::format_tag::x
              },
              cpu_engine, inout->data());

  auto relu_descriptor = mkldnn::eltwise_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::eltwise_relu,
          inout_memory.get_desc(), 0.0f, 0.0f);
  auto relu_primitive = mkldnn::eltwise_forward::primitive_desc(
                  relu_descriptor, cpu_engine);

  auto relu = mkldnn::eltwise_forward(relu_primitive);
  relu.execute(cpu_stream, {
    {MKLDNN_ARG_SRC, inout_memory},
    {MKLDNN_ARG_DST, inout_memory},
  });
}
