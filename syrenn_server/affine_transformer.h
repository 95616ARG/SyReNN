#ifndef SYRENN_SYRENN_SERVER_AFFINE_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_AFFINE_TRANSFORMER_H_

#include <string>
#include <vector>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/transformer.h"

// Base (abstract) class for layers that are affine (i.e., f(cx) = cf(x) and
// f(x + y) = f(x) + f(y) for any x, y in their domain). Provides efficient
// implementations of the abstract transformers. Child classes need to
// implement Compute(...) and size_t out_size(size_t in_size) as defined on
// LayerTransformer.
//
// Child classes:
// - FullyConnectedTransformer
// - Conv2DTransformer
// - NormalizeTransformer
// - AveragePoolTransformer
class AffineTransformer : public LayerTransformer {
 public:
  void TransformUPolytope(UPolytope *inout) const override;
  std::vector<double> ProposeLineEndpoints(
      const SegmentedLine &line) const override;

  std::string layer_type() const override { return "Affine"; };
};

#endif  // SYRENN_SYRENN_SERVER_AFFINE_TRANSFORMER_H_
