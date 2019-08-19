#ifndef SYRENN_SYRENN_SERVER_RELU_MAXPOOL_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_RELU_MAXPOOL_TRANSFORMER_H_

#include <string>
#include <vector>
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/maxpool_transformer.h"
#include "syrenn_server/transformer.h"

// Transformer for fused ReLU + MaxPool layers. This is automatically
// substituted for sequential ReLU/MaxPool layers in server.cc.
class ReLUMaxPoolTransformer : public MaxPoolTransformer {
 public:
  using MaxPoolTransformer::MaxPoolTransformer;

  explicit ReLUMaxPoolTransformer(const MaxPoolTransformer &other)
      : MaxPoolTransformer(other) {}

  virtual void process_window(
      // The windows in (H, W) format.
      const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &from_window,
      const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &to_window,
      // The preimage-ratios for each endpoint.
      double from_ratio, double to_ratio,
      // The set to place the endpoints in.
      tbb::concurrent_set<double> *endpoints) const;

  void Compute(RMMatrixXf *inout) const;
  std::string layer_type() const override {
    return "ReLUMaxPool";
  };
};

#endif  // SYRENN_SYRENN_SERVER_RELU_MAXPOOL_TRANSFORMER_H_
