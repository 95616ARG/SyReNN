#include "syrenn_server/strided_window_data.h"
#include "mkldnn.hpp"

size_t StridedWindowData::in_height() const {
  return in_height_;
}

size_t StridedWindowData::in_width() const {
  return in_width_;
}

size_t StridedWindowData::padded_in_height() const {
  return in_height_ + (2 * pad_height_);
}

size_t StridedWindowData::padded_in_width() const {
  return in_width_ + (2 * pad_width_);
}

size_t StridedWindowData::in_channels() const {
  return in_channels_;
}

size_t StridedWindowData::in_size() const {
  return in_height_ * in_width_ * in_channels_;
}

size_t StridedWindowData::window_height() const {
  return window_height_;
}

size_t StridedWindowData::window_width() const {
  return window_width_;
}

size_t StridedWindowData::out_channels() const {
  return out_channels_;
}

size_t StridedWindowData::stride_height() const {
  return stride_height_;
}

size_t StridedWindowData::stride_width() const {
  return stride_width_;
}

size_t StridedWindowData::pad_height() const {
  return pad_height_;
}

size_t StridedWindowData::pad_width() const {
  return pad_width_;
}

size_t StridedWindowData::out_height() const {
  return 1 +
    (((2 * pad_height_) + in_height_ - window_height_) / stride_height_);
}

size_t StridedWindowData::out_width() const {
  return 1 +
    (((2 * pad_width_) + in_width_ - window_width_) / stride_width_);
}

size_t StridedWindowData::out_size() const {
  return out_channels_ * out_height() * out_width();
}

mkldnn::memory::dims StridedWindowData::mkl_input_dims(size_t batch) const {
  // NOTE: MKL reads the dimension sizes in NCHW even though the layout we
  // store it in is NHWC
  return {
    static_cast<int>(batch), static_cast<int>(in_channels_),
    static_cast<int>(in_height_), static_cast<int>(in_width_)
  };
}

mkldnn::memory::dims StridedWindowData::mkl_filter_dims() const {
  // NOTE: Ditto here, MKL reads OIHW even though we store HWIO.
  return {
    static_cast<int>(out_channels_), static_cast<int>(in_channels_),
    static_cast<int>(window_height_), static_cast<int>(window_width_)
  };
}

mkldnn::memory::dims StridedWindowData::mkl_window_dims() const {
  // NOTE: Ditto here, MKL reads OIHW even though we store HWIO.
  return {
    static_cast<int>(window_height_), static_cast<int>(window_width_)
  };
}

mkldnn::memory::dims StridedWindowData::mkl_bias_dims() const {
  return {
    static_cast<int>(out_channels_)
  };
}

mkldnn::memory::dims StridedWindowData::mkl_output_dims(size_t batch) const {
  return {
    static_cast<int>(batch), static_cast<int>(out_channels_),
    static_cast<int>(out_height()), static_cast<int>(out_width())
  };
}

mkldnn::memory::dims StridedWindowData::mkl_stride_dims() const {
  return {
    static_cast<int>(stride_height_), static_cast<int>(stride_width_)
  };
}

mkldnn::memory::dims StridedWindowData::mkl_pad_dims() const {
  return {
    static_cast<int>(pad_height_), static_cast<int>(pad_width_)
  };
}
