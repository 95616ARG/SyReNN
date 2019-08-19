#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/conv2d_transformer.h"
#include "syrenn_server/strided_window_data.h"
#include "mkldnn.hpp"

Conv2DTransformer::Conv2DTransformer(const RMMatrixXf &filters,
                                     const RMVectorXf &biases,
                                     const StridedWindowData &window_data)
    : filters_(filters), biases_(biases), window_data_(window_data) {}

size_t Conv2DTransformer::out_size(size_t in_size) const {
  return window_data_.out_size();
}

std::unique_ptr<LayerTransformer> Conv2DTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_conv2d_data()) {
    return nullptr;
  }
  const auto &conv2d_data = layer.conv2d_data();
  const auto window_data = StridedWindowData::Deserialize(
      conv2d_data.window_data());

  // NOTE we are assuming layouts here
  auto filter_rows = window_data.window_height() * window_data.window_width();
  Eigen::Map<const RMMatrixXf> filters(
          conv2d_data.filters().data(),
          filter_rows,
          conv2d_data.filters_size() / filter_rows);
  Eigen::Map<const RMVectorXf> biases(
                  conv2d_data.biases().data(),
                  conv2d_data.biases_size());

  return std::unique_ptr<LayerTransformer>(
            new Conv2DTransformer(filters, biases, window_data));
}

void Conv2DTransformer::Compute(RMMatrixXf *inout) const {
  // input = (N, Hi*Wi*Ci)
  // this->filters_ = (Hf*Wf, Ci*Co)
  // this->biases_ = (Co,)

  mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
  mkldnn::stream cpu_stream(cpu_engine);

  int batch = inout->rows();

  mkldnn::memory::dims input_dims = window_data_.mkl_input_dims(batch);
  mkldnn::memory::dims filter_dims = window_data_.mkl_filter_dims();
  mkldnn::memory::dims bias_dims = window_data_.mkl_bias_dims();
  // NOTE: Ditto, MKL reads NCHW, we store NHWC
  mkldnn::memory::dims output_dims = window_data_.mkl_output_dims(batch);
  mkldnn::memory::dims strides = window_data_.mkl_stride_dims();
  mkldnn::memory::dims padding = window_data_.mkl_pad_dims();
  // Make mutable, float32 copies of the data for passing into MKL.
  RMMatrixXf filters_copy = filters_;
  RMVectorXf biases_copy = biases_;
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
  auto filters_memory =
      mkldnn::memory(
          {
              { filter_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::hwio
          }, cpu_engine, filters_copy.data());
  auto biases_memory =
      mkldnn::memory(
          {
              { bias_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::x
          }, cpu_engine, biases_copy.data());
  auto output_memory =
      mkldnn::memory(
          {
              { output_dims },
              mkldnn::memory::data_type::f32,
              mkldnn::memory::format_tag::nhwc
          }, cpu_engine, output_data.data());

  // Memory description for the convolution, with no specified format.
  auto input_descriptor =
      mkldnn::memory::desc(
          { input_dims },
          mkldnn::memory::data_type::f32,
          mkldnn::memory::format_tag::any);
  auto biases_descriptor =
      mkldnn::memory::desc(
          { bias_dims },
          mkldnn::memory::data_type::f32,
          mkldnn::memory::format_tag::any);
  auto filters_descriptor =
      mkldnn::memory::desc(
          { filter_dims },
          mkldnn::memory::data_type::f32,
          mkldnn::memory::format_tag::any);
  auto output_descriptor =
      mkldnn::memory::desc(
          { output_dims },
          mkldnn::memory::data_type::f32,
          mkldnn::memory::format_tag::any);

  // Conv2D
  auto convolution_descriptor = mkldnn::convolution_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::convolution_direct, input_descriptor,
          filters_descriptor, biases_descriptor, output_descriptor, strides,
          padding, padding);
  auto convolution_primitive = mkldnn::convolution_forward::primitive_desc(
                  convolution_descriptor, cpu_engine);

  // Reorder data & weights if layout requested by convolution is different
  // from NHWC/HWIO.
  auto convolution_input_memory = input_memory;
  if (convolution_primitive.src_desc() != input_memory.get_desc()) {
    convolution_input_memory = mkldnn::memory(convolution_primitive.src_desc(),
                                              cpu_engine);
    auto reorder = mkldnn::reorder(input_memory, convolution_input_memory);
    reorder.execute(cpu_stream, {
        {MKLDNN_ARG_FROM, input_memory},
        {MKLDNN_ARG_TO, convolution_input_memory},
    });
  }

  auto convolution_filters_memory = filters_memory;
  if (convolution_primitive.weights_desc() != filters_memory.get_desc()) {
    convolution_filters_memory
        = mkldnn::memory(convolution_primitive.weights_desc(), cpu_engine);
    auto reorder = mkldnn::reorder(filters_memory, convolution_filters_memory);
    reorder.execute(cpu_stream, {
        {MKLDNN_ARG_FROM, filters_memory},
        {MKLDNN_ARG_TO, convolution_filters_memory},
    });
  }

  auto convolution_output_memory =
      mkldnn::memory(convolution_primitive.dst_desc(), cpu_engine);

  auto convolution = mkldnn::convolution_forward(convolution_primitive);
  convolution.execute(cpu_stream, {
    { MKLDNN_ARG_SRC, convolution_input_memory },
    { MKLDNN_ARG_WEIGHTS, convolution_filters_memory },
    { MKLDNN_ARG_BIAS, biases_memory },
    { MKLDNN_ARG_DST, convolution_output_memory }
  });

  auto reorder = mkldnn::reorder(convolution_output_memory, output_memory);
  reorder.execute(cpu_stream, {
    {MKLDNN_ARG_FROM, convolution_output_memory},
    {MKLDNN_ARG_TO, output_memory},
  });

  inout->swap(output_data);
  output_data.resize(0, 0);
}
