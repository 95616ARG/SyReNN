#include "gtest/gtest.h"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/pwl_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

// Relu but only in the first coefficient.
class ReLUFirstTransformer : public PWLTransformer {
 public:
  void Compute(RMMatrixXf *inout) const override {
    for (int i = 0; i < inout->rows(); i++) {
      if ((*inout)(i, 0) < 0.0) {
        (*inout)(i, 0) = 0.0;
      }
    }
  }
  size_t out_size(size_t in_size) const override { return in_size; }

 protected:
  size_t n_piece_faces(size_t dims) const override {
    return 1;
  }
  double CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                       Eigen::Ref<const RMVectorXf> to,
                       const size_t face) const override {
    EXPECT_EQ(face, 0ul);
    return -from(0) / (to(0) - from(0));
  }
  int PointSign(Eigen::Ref<const RMVectorXf> point,
                const size_t face) const override {
    EXPECT_EQ(face, 0ul);
    if (point(0) == 0) {
      return 0;
    }
    return point(0) > 0 ? +1 : -1;
  }
};

TEST(PWLTransformer, TransformLine) {
  RMVectorXf startpoint(2);
  startpoint << -1.0, 3.0;
  RMVectorXf endpoint(2);
  endpoint << 3.0, -1.0;

  SegmentedLine line(startpoint, endpoint);

  ReLUFirstTransformer transformer;
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(3ul, line.Size());

  RMMatrixXf post_points(3, 2);
  post_points << 0.0, 3.0,
                 0.0, 2.0,
                 3.0, -1.0;
  std::vector<double> post_distances {0.0, 1.0 / 4.0, 1.0};

  for (size_t i = 0; i < post_distances.size(); i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
    EXPECT_EQ(post_points.row(i), line.points().row(i));
  }
}

TEST(PWLTransformer, TransformPlanes) {
  RMMatrixXf vertices(3, 2);
  vertices << 1.0, 1.0,
              3.0, -1.0,
              2.0, -2.0;
  RMMatrixXf truth_vertices = vertices;
  RMMatrixXf truth_combinations(3, 3);
  truth_combinations.setIdentity();

  UPolytope polytope(&vertices, 2, {{0, 1, 2}});

  ReLUFirstTransformer transformer;
  transformer.TransformUPolytope(&polytope);

  EXPECT_EQ(polytope.vertices(), truth_vertices);
  EXPECT_EQ(polytope.combinations(), truth_combinations);
}
