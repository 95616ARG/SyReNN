#include <iostream>
#include <vector>
#include "eigen3/Eigen/Dense"
#include "syrenn_server/affine_transformer.h"

std::vector<double> AffineTransformer::ProposeLineEndpoints(
    const SegmentedLine &line) const {
  return {};
}

void AffineTransformer::TransformUPolytope(UPolytope *inout) const {
  Compute(&inout->vertices());
}
