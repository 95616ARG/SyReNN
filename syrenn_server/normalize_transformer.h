#ifndef SYRENN_SYRENN_SERVER_NORMALIZE_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_NORMALIZE_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/affine_transformer.h"

// Transformer for normalize layers. Means/standard_deviations are assumed to
// broadcast with the most-minor dimension of the input (eg. channels for
// images).
class NormalizeTransformer : public AffineTransformer {
 public:
  NormalizeTransformer(const RMVectorXf &means,
                       const RMVectorXf &standard_deviations);
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  void Compute(RMMatrixXf *inout) const;
  std::string layer_type() const override { return "Normalize"; }
  size_t out_size(size_t in_size) const override { return in_size; }

 private:
  const RMVectorXf means_;
  const RMVectorXf standard_deviations_;
};

#endif  // SYRENN_SYRENN_SERVER_NORMALIZE_TRANSFORMER_H_
