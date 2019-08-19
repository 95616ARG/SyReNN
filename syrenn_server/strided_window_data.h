#ifndef SYRENN_SYRENN_SERVER_STRIDED_WINDOW_DATA_H_
#define SYRENN_SYRENN_SERVER_STRIDED_WINDOW_DATA_H_

#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "mkldnn.hpp"

// Helper class for all layers that use strided windows.
//
// Used by:
// - Conv2DTransformer
// - MaxPoolTransformer
// - AveragePoolTransformer
class StridedWindowData {
 public:
  StridedWindowData(size_t in_height, size_t in_width, size_t in_channels,
                    size_t window_height, size_t window_width,
                    size_t out_channels, size_t stride_height,
                    size_t stride_width, size_t pad_height, size_t pad_width)
      :   // Input info
        in_height_(in_height), in_width_(in_width), in_channels_(in_channels),
          // Filter info
        window_height_(window_height), window_width_(window_width),
        out_channels_(out_channels),
          // Stride info
        stride_height_(stride_height), stride_width_(stride_width),
          // Padding info
        pad_height_(pad_height), pad_width_(pad_width) {}

  static StridedWindowData Deserialize(
      const syrenn_server::StridedWindowData &data) {
    return StridedWindowData(
        data.in_height(), data.in_width(), data.in_channels(),
        data.window_height(), data.window_width(), data.out_channels(),
        data.stride_height(), data.stride_width(), data.pad_height(),
        data.pad_width());
  }

  size_t in_height() const;
  size_t in_width() const;
  size_t padded_in_height() const;
  size_t padded_in_width() const;
  size_t in_channels() const;
  size_t in_size() const;

  size_t window_height() const;
  size_t window_width() const;
  size_t out_channels() const;

  size_t stride_height() const;
  size_t stride_width() const;

  size_t pad_height() const;
  size_t pad_width() const;

  size_t out_height() const;
  size_t out_width() const;
  size_t out_size() const;

  mkldnn::memory::dims mkl_input_dims(size_t batch) const;
  mkldnn::memory::dims mkl_filter_dims() const;
  mkldnn::memory::dims mkl_window_dims() const;
  mkldnn::memory::dims mkl_bias_dims() const;
  mkldnn::memory::dims mkl_output_dims(size_t batch) const;
  mkldnn::memory::dims mkl_stride_dims() const;
  mkldnn::memory::dims mkl_pad_dims() const;

 protected:
  const size_t in_height_;
  const size_t in_width_;
  const size_t in_channels_;

  const size_t window_height_;
  const size_t window_width_;
  const size_t out_channels_;

  const size_t stride_height_;
  const size_t stride_width_;

  const size_t pad_height_;
  const size_t pad_width_;
};

#endif  // SYRENN_SYRENN_SERVER_STRIDED_WINDOW_DATA_H_
