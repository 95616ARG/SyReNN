#include <memory>
#include "syrenn_server/hard_tanh_transformer.h"
#include "eigen3/Eigen/Dense"

std::unique_ptr<LayerTransformer> HardTanhTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_hard_tanh_data()) {
    return nullptr;
  }
  return std::unique_ptr<HardTanhTransformer>(new HardTanhTransformer());
}

size_t HardTanhTransformer::n_piece_faces(size_t dims) const {
  return 2 * dims;
}

double HardTanhTransformer::CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                                          Eigen::Ref<const RMVectorXf> to,
                                          const size_t face) const {
  int dim = face / 2;
  double cutoff = (face % 2 == 1) ? 1.0 : -1.0;
  return (cutoff - from(dim)) / (to(dim) - from(dim));
}

int HardTanhTransformer::PointSign(Eigen::Ref<const RMVectorXf> point,
                                   const size_t face) const {
  size_t dim = face / 2;

  if (face % 2 == 1) {
    if (point(dim) == 1.0) {
      return 0;
    }
    return point(dim) > 1.0 ? +1 : -1;
  }

  if (point(dim) == -1.0) {
    return 0;
  }
  return point(dim) > -1.0 ? +1 : -1;
}

void HardTanhTransformer::Compute(RMMatrixXf *inout) const {
  (*inout) = inout->cwiseMax(-1.0).cwiseMin(1.0);
}
