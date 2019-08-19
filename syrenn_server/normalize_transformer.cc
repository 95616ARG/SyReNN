#include <iostream>
#include <memory>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/normalize_transformer.h"

NormalizeTransformer::NormalizeTransformer(const RMVectorXf &means,
                                   const RMVectorXf &standard_deviations)
    : means_(means), standard_deviations_(standard_deviations) {}

std::unique_ptr<LayerTransformer> NormalizeTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_normalize_data()) {
    return nullptr;
  }
  Eigen::Map<const RMVectorXf> means(
                  layer.normalize_data().means().data(),
                  layer.normalize_data().means_size());
  Eigen::Map<const RMVectorXf> standard_deviations(
                  layer.normalize_data().standard_deviations().data(),
                  layer.normalize_data().standard_deviations_size());

  return std::unique_ptr<LayerTransformer>(
            new NormalizeTransformer(means, standard_deviations));
}

void NormalizeTransformer::Compute(RMMatrixXf *inout) const {
  // input will be (N, H*W*C)
  // We reshape to (N*H*W, C)
  unsigned int hw = inout->cols() / means_.size();
  size_t batch = inout->rows();
  inout->resize(batch * hw, means_.size());

  inout->array().rowwise() -= means_.array();
  inout->array().rowwise() /= standard_deviations_.array();

  inout->resize(batch, hw * means_.size());
}
