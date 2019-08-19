#ifndef SYRENN_SYRENN_SERVER_FULLYCONNECTED_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_FULLYCONNECTED_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/affine_transformer.h"

// Transformer for Fully-Connected layers.
class FullyConnectedTransformer : public AffineTransformer {
 public:
  FullyConnectedTransformer(const RMMatrixXf &weights,
                            const RMVectorXf &biases);
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  void Compute(RMMatrixXf *inout) const;
  std::string layer_type() const override { return "Fully-Connected"; };
  size_t out_size(size_t in_size) const { return biases_.size(); }
 private:
  const RMMatrixXf weights_;
  const RMVectorXf biases_;
};

#endif  // SYRENN_SYRENN_SERVER_FULLYCONNECTED_TRANSFORMER_H_
