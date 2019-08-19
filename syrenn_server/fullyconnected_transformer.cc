#include <iostream>
#include <memory>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/fullyconnected_transformer.h"

FullyConnectedTransformer::FullyConnectedTransformer(const RMMatrixXf &weights,
                                                     const RMVectorXf &biases)
    : weights_(weights), biases_(biases) {}

std::unique_ptr<LayerTransformer> FullyConnectedTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_fullyconnected_data()) {
    return nullptr;
  }
  int n_out = layer.fullyconnected_data().biases_size();
  int n_in = layer.fullyconnected_data().weights_size() / n_out;

  // NOTE we are assuming row-major, (in, out) here
  Eigen::Map<const RMMatrixXf> weights(
                  layer.fullyconnected_data().weights().data(),
                  n_in, n_out);
  Eigen::Map<const RMVectorXf> biases(
                  layer.fullyconnected_data().biases().data(), n_out);

  // If we can figure out how to get Bazel to build
  // with C++14, we could also use make_unique
  return std::unique_ptr<LayerTransformer>(
      new FullyConnectedTransformer(weights, biases));
}

void FullyConnectedTransformer::Compute(RMMatrixXf *inout) const {
  // TODO(masotoud): Use noalias here?
  (*inout) *= weights_;
  inout->rowwise() += biases_;
}
