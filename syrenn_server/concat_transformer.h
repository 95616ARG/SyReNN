#ifndef SYRENN_SYRENN_SERVER_CONCAT_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_CONCAT_TRANSFORMER_H_

#include <string>
#include <memory>
#include <vector>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/transformer.h"

enum class ConcatAlong {
  CHANNELS,
  FLAT,
};

// This layer handles concatenations of other layers, i.e.:
//
// y = Concat(3 * x, 2 * x) = [3*x_1, ..., 3*x_n, 2*x_1, ..., 2*x_n]
//
// Note that the Concat layer transformer logically encompases its "feeder"
// layers, and can be thought of as a "diamond" if that helps. In otherwords,
// the above example is more accurately:
//
// y = Concat([x |-> 3 * x, x |-> 2 * x])(x) =
//        [3*x_1, ..., 3*x_n, 2*x_1, ..., 2*x_n]
//
// If @concat_along = ConcatAlong::CHANNELS, the inputs will be concatenated
// along the channel dimension (i.e., all inputs should have the same
// height/width shape and *all input transformers should be Conv2DLayers*).
//
// If @concat_along = ConcatAlong::FLAT, the inputs will be flattened before
// concatenation (no restriction on input transformers).
class ConcatTransformer : public LayerTransformer {
 public:
  ConcatTransformer(
      std::vector<std::unique_ptr<LayerTransformer>> *input_transformers,
      const ConcatAlong &concat_along);
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  void Compute(RMMatrixXf *inout) const;
  size_t out_size(size_t in_size) const override;
  std::string layer_type() const override { return "Concat"; };

  std::vector<double> ProposeLineEndpoints(
      const SegmentedLine &line) const override;

 private:
  std::vector<std::unique_ptr<LayerTransformer>> input_transformers_;
  const ConcatAlong concat_along_;
};

#endif  // SYRENN_SYRENN_SERVER_CONCAT_TRANSFORMER_H_
