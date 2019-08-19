#ifndef SYRENN_SYRENN_SERVER_HARD_TANH_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_HARD_TANH_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/pwl_transformer.h"

// Transformer for Hard Tanh layers.
class HardTanhTransformer : public PWLTransformer {
 public:
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  void Compute(RMMatrixXf *inout) const override;
  std::string layer_type() const override { return "HardTanh"; };
  size_t out_size(size_t in_size) const override { return in_size; }

 protected:
  size_t n_piece_faces(size_t dims) const override;
  double CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                       Eigen::Ref<const RMVectorXf> to,
                       const size_t face) const override;
  int PointSign(Eigen::Ref<const RMVectorXf> point,
                const size_t face) const override;
};

#endif  // SYRENN_SYRENN_SERVER_HARD_TANH_TRANSFORMER_H_
