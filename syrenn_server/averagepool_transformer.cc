#include <memory>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/averagepool_transformer.h"
#include "syrenn_server/strided_window_data.h"
#include "mkldnn.hpp"

AveragePoolTransformer::AveragePoolTransformer(
    const StridedWindowData &window_data)
    : window_data_(window_data) {}

size_t AveragePoolTransformer::out_size(size_t in_size) const {
  return window_data_.out_size();
}

std::unique_ptr<LayerTransformer> AveragePoolTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_averagepool_data()) {
    return nullptr;
  }
  const auto &averagepool_data = layer.averagepool_data();
  const auto window_data = StridedWindowData::Deserialize(
      averagepool_data.window_data());

  return std::unique_ptr<LayerTransformer>(
        new AveragePoolTransformer(window_data));
}

void AveragePoolTransformer::Compute(RMMatrixXf *inout) const {
  // Modified from
  // https://github.com/intel/mkl-dnn/blob/mnt-v0/examples/simple_net.cpp
  // See conv2d_transformer.cc for more.

  mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
  mkldnn::stream cpu_stream(cpu_engine);

  int batch = inout->rows();

  mkldnn::memory::dims input_dims = window_data_.mkl_input_dims(batch);
  mkldnn::memory::dims window_dims = window_data_.mkl_window_dims();
  mkldnn::memory::dims strides = window_data_.mkl_stride_dims();
  mkldnn::memory::dims output_dims = window_data_.mkl_output_dims(batch);
  mkldnn::memory::dims padding = window_data_.mkl_pad_dims();

  // Initialize an output buffer for MKL to use
  RMMatrixXf output_data(batch, window_data_.out_size());
  output_data.setZero();

  // MKL memory references to the above buffers
  auto input_memory =
      mkldnn::memory(
          {
              { input_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::nhwc
          }, cpu_engine, inout->data());
  auto output_memory =
      mkldnn::memory(
          {
              { output_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::nhwc
          }, cpu_engine, output_data.data());

  // create an averagepool:
  // https://intel.github.io/mkl-dnn/cpu_cnn_inference_f32_8c-example.html#a41
  auto pool_descriptor = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::pooling_avg_include_padding,
          input_memory.get_desc(), output_memory.get_desc(),
          strides, window_dims, padding, padding);
  auto pool_primitive = mkldnn::pooling_forward::primitive_desc(
          pool_descriptor, cpu_engine);

  auto pool = mkldnn::pooling_forward(pool_primitive);
  pool.execute(cpu_stream, {
    {MKLDNN_ARG_SRC, input_memory},
    {MKLDNN_ARG_DST, output_memory},
  });

  inout->swap(output_data);
}
