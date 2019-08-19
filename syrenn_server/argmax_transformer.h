#ifndef SYRENN_SYRENN_SERVER_ARGMAX_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_ARGMAX_TRANSFORMER_H_

#include <memory>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/pwl_transformer.h"
#include "syrenn_server/transformer.h"

// Transformer for an ArgMax layer. This is mostly used as a helper method in
// the front-end to determine classification of lines/planes (see
// helpers/classify_{lines, planes}.py).
//
// *NOTE:* using this in a call to the line/plane transformer with
// include_post=True may not do what you expect, because the argmax at each of
// the endpoints will be ill-defined (as the endpoints are where the argmax
// change). See the Helpers for examples of how to use this correctly ---
// namely, with include_post=False. This is the only non-continuous function
// that has a transformer for it in this repository.
class ArgMaxTransformer : public PWLTransformer {
 public:
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  void Compute(RMMatrixXf *inout) const override;
  std::string layer_type() const override { return "ArgMax"; };
  size_t out_size(size_t in_size) const { return 1; }
 protected:
  size_t n_piece_faces(size_t dims) const override;
  double CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                       Eigen::Ref<const RMVectorXf> to,
                       const size_t face) const override;
  bool IsFaceActive(Eigen::Ref<const RMVectorXf> from,
                    Eigen::Ref<const RMVectorXf> to,
                    const size_t face) const override;
  int PointSign(Eigen::Ref<const RMVectorXf> point,
                const size_t face) const override;
};

#endif  // SYRENN_SYRENN_SERVER_ARGMAX_TRANSFORMER_H_
