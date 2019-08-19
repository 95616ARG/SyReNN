#ifndef SYRENN_SYRENN_SERVER_CONV2D_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_CONV2D_TRANSFORMER_H_

#include <memory>
#include <string>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/segmented_line.h"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/affine_transformer.h"
#include "syrenn_server/strided_window_data.h"

// Transformer for 2D convolution layers.
class Conv2DTransformer : public AffineTransformer {
 public:
  // We expect filters.shape = (height*width, in_channels*out_channels)
  // We expect biases.shape = (out_channels,)
  Conv2DTransformer(const RMMatrixXf &filters,
                    const RMVectorXf &biases,
                    const StridedWindowData &window_data);
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  // input MUST be row-major
  // input.shape = (N, Hi*Wi*Ci)
  void Compute(RMMatrixXf *inout) const;
  unsigned int out_channels() const { return window_data_.out_channels(); }
  size_t out_size(size_t in_size) const override;
  std::string layer_type() const override { return "Conv2D"; }

 private:
  const RMMatrixXf filters_;
  const RMVectorXf biases_;
  const StridedWindowData window_data_;
};

#endif  // SYRENN_SYRENN_SERVER_CONV2D_TRANSFORMER_H_
