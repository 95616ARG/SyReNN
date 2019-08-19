#include <iostream>
#include <unordered_set>
#include "syrenn_server/relu_maxpool_transformer.h"
#include "syrenn_server/relu_transformer.h"
#include "mkldnn.hpp"

double sign_change_ratio(double from_ratio, double to_ratio,
                         double from_value, double to_value) {
  double local_ratio = -from_value / (to_value - from_value);
  return from_ratio + (local_ratio * (to_ratio - from_ratio));
}

void ReLUMaxPoolTransformer::process_window(
        const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &from_window,
        const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &to_window,
        double from_ratio, double to_ratio,
        tbb::concurrent_set<double> *endpoints) const {
  // Find the existing maxes of the windows.
  WindowPixel from_argmax, to_argmax;
  double from_max = from_window.maxCoeff(&from_argmax.h, &from_argmax.w);
  double to_max = to_window.maxCoeff(&to_argmax.h, &to_argmax.w);

  // We check these special cases to avoid having to copy from_window below
  // in a number of common cases.
  if ((from_max < 0.0 && to_max < 0.0) ||
          (from_argmax == to_argmax && from_max > 0.0 && to_max > 0.0)) {
    // A or B, where A = the post-maxpool node is always ReLU'd away and B =
    // post-maxpool node is just linear (max is consistent and ReLU does
    // nothing).
    // In either of these cases, it is linear and we don't need to add any new
    // endpoints.
    return;
  }
  if (from_argmax == to_argmax) {
    // In this case, all we need to worry about is the sign change of the
    // max element (note if the sign didn't change and this condition was
    // met, it would have been caught above).
    double ratio = sign_change_ratio(from_ratio, to_ratio, from_max, to_max);
    endpoints->insert(ratio);
    return;
  }

  std::unordered_set<WindowPixel> old_argmaxes{from_argmax};

  RMMatrixXf delta = to_window - from_window;
  RMMatrixXf crossing_distances(from_window.rows(), from_window.cols());
  double last_ratio = 0.0;

  if (from_max < 0.0) {
    // We can skip straight to where the first dimension hits 0
    // 0 = (S + tD)[i]
    // t = -S[i] / D[i]
    crossing_distances = -from_window.array() / delta.array();
    for (Eigen::Index row = 0; row < crossing_distances.rows(); row++) {
    for (Eigen::Index col = 0; col < crossing_distances.cols(); col++) {
      if (crossing_distances(row, col) < 0.0) {
          crossing_distances(row, col) = 2.0;
      }
    }
    }

    last_ratio = crossing_distances.minCoeff(&from_argmax.h, &from_argmax.w);

    double global_ratio = from_ratio + (last_ratio * (to_ratio - from_ratio));
    endpoints->insert(global_ratio);
  }

  // Loop over linear regions.
  while (old_argmaxes.size() < static_cast<size_t>(from_window.size())) {
    // Find the start of the closest linear region.
    WindowPixel best_arg = to_argmax;
    double best_ratio = 1.0;
    for (Eigen::Index row = 0; row < from_window.rows(); row++) {
    for (Eigen::Index col = 0; col < from_window.cols(); col++) {
      WindowPixel pixel(row, col);
      if (old_argmaxes.count(pixel) == 1) {
        continue;
      }
      // "Local" ratio between from_window and to_window.
      // (S + tD)[n] = (S + tD)[i]
      // S[n] - S[i] = t(D[i] - D[n])
      // t = (S[n] - S[i]) / (D[i] - D[n])
      double crossing_ratio =
        (from_window(row, col) - from_window(from_argmax.h, from_argmax.w)) /
        (delta(from_argmax.h, from_argmax.w) - delta(row, col));
      if (crossing_ratio > last_ratio && crossing_ratio < best_ratio) {
        best_arg = pixel;
        best_ratio = crossing_ratio;
      }
    }
    }

    if (best_ratio == 1.0) {
      break;
    }

    // Here we have to check for the case where it goes positive -> negative.
    double from_value = from_window(best_arg.h, best_arg.w);
    double to_value = to_window(best_arg.h, best_arg.w);
    double value = from_value + (best_ratio * (to_value - from_value));
    if (value <= 0.0) {
      double ratio = sign_change_ratio(from_ratio, to_ratio,
                                       from_value, to_value);
      endpoints->insert(ratio);
      return;
    }

    from_argmax = best_arg;
    double global_ratio = from_ratio + (best_ratio * (to_ratio - from_ratio));
    // Finally, we insert the crossing distance.
    endpoints->insert(global_ratio);
    last_ratio = best_ratio;
    if (best_arg == to_argmax) {
      break;
    }
  }

  if (to_max < 0) {
    double ratio = sign_change_ratio(from_ratio, to_ratio,
                                     from_window(to_argmax.h, to_argmax.w),
                                     to_max);
    endpoints->insert(ratio);
  }
}

void ReLUMaxPoolTransformer::Compute(RMMatrixXf *inout) const {
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

  // MaxPool
  auto pool_descriptor = mkldnn::pooling_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::pooling_max,
          input_memory.get_desc(), output_memory.get_desc(),
          strides, window_dims, padding, padding);
  auto pool_primitive = mkldnn::pooling_forward::primitive_desc(
          pool_descriptor, cpu_engine);
  auto pool = mkldnn::pooling_forward(pool_primitive);
  pool.execute(cpu_stream, {
    {MKLDNN_ARG_SRC, input_memory},
    {MKLDNN_ARG_DST, output_memory},
  });

  // ReLU
  auto relu_descriptor = mkldnn::eltwise_forward::desc(
          mkldnn::prop_kind::forward_inference,
          mkldnn::algorithm::eltwise_relu,
          output_memory.get_desc(), 0.0f, 0.0f);
  auto relu_primitive = mkldnn::eltwise_forward::primitive_desc(
                  relu_descriptor, cpu_engine);

  auto relu = mkldnn::eltwise_forward(relu_primitive);
  relu.execute(cpu_stream, {
    {MKLDNN_ARG_SRC, output_memory},
    {MKLDNN_ARG_DST, output_memory},
  });

  inout->swap(output_data);
  output_data.resize(0, 0);
}
