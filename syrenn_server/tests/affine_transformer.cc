#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/affine_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

class DoubleTransformer : public AffineTransformer {
  public:
   void Compute(RMMatrixXf *inout) const override {
     *inout = 2.0 * (*inout);
   }
   size_t out_size(size_t in_size) const override {
     return in_size;
   }
};

TEST(AffineTransformer, TransformLine) {
  RMVectorXf startpoint(2);
  startpoint << -1.0, 3.0;
  RMVectorXf endpoint(2);
  endpoint << 3.0, -1.0;

  SegmentedLine line(startpoint, endpoint);

  DoubleTransformer transformer;
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(2ul, line.Size());

  RMMatrixXf post_points(2, 2);
  post_points << -2.0, 6.0,
                 6.0, -2.0;
  std::vector<double> post_distances {0.0, 1.0};

  for (size_t i = 0; i < post_distances.size(); i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
    EXPECT_EQ(post_points.row(i), line.points().row(i));
  }
}

TEST(AffineTransformer, TransformPlanes) {
  RMMatrixXf vertices(4, 2);
  vertices << 1.0, 1.0,
              -1.0, 1.0,
              -1.0, -1.0,
              1.0, -1.0;
  RMMatrixXf truth_vertices = vertices * 2.0;
  RMMatrixXf truth_combinations(4, 4);
  truth_combinations.setIdentity();

  UPolytope polytope(&vertices, 2, {{0, 1, 2, 3}});

  DoubleTransformer transformer;
  transformer.TransformUPolytope(&polytope);

  EXPECT_EQ(polytope.vertices(), truth_vertices);
  EXPECT_EQ(polytope.combinations(), truth_combinations);
}
