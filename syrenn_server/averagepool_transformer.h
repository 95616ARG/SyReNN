#ifndef SYRENN_SYRENN_SERVER_AVERAGEPOOL_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_AVERAGEPOOL_TRANSFORMER_H_

#include <memory>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/affine_transformer.h"
#include "syrenn_server/strided_window_data.h"

// Represents an AveragePool layer, where only the mean of each window is
// passed to the next layer.
// NOTE: As with PyTorch, AveragePoolTransformer *DOES* count the padded zeros
// as values to be averaged with.
class AveragePoolTransformer : public AffineTransformer {
 public:
  explicit AveragePoolTransformer(const StridedWindowData &window_data);

  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);
  void Compute(RMMatrixXf *inout) const;
  size_t out_size(size_t in_size) const override;

 private:
  const StridedWindowData &window_data_;
};

#endif  // SYRENN_SYRENN_SERVER_AVERAGEPOOL_TRANSFORMER_H_
