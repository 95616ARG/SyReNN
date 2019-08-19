#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/argmax_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(ArgMaxTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto argmax_data = serialized.mutable_argmax_data();
  EXPECT_EQ(!argmax_data, false);
  auto deserialized = ArgMaxTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(ArgMaxTransformer, Compute) {
  const size_t n_points = 1024, dimensions = 4096;

  RMMatrixXf batch(n_points, dimensions);
  batch.setRandom();
  RMMatrixXf truth(n_points, 1);
  for (size_t row = 0; row < n_points; row++) {
    batch.row(row).maxCoeff(&truth(row, 0));
  }

  ArgMaxTransformer transformer;
  transformer.Compute(&batch);
  EXPECT_EQ(batch, truth);
}

TEST(ArgMaxTransformer, out_size) {
  ArgMaxTransformer transformer;
  EXPECT_EQ(transformer.out_size(1024), 1ul);
}

TEST(ArgMaxTransformer, TransformLine) {
  RMVectorXf startpoint(2);
  startpoint << 1.0, 0.0;
  RMVectorXf endpoint(2);
  endpoint << 1.0, 2.0;

  SegmentedLine line(startpoint, endpoint);

  ArgMaxTransformer transformer;
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(3ul, line.Size());

  // We don't use/check the post-points for ArgMaxTransformer, as the endpoints
  // are exactly where argmax is ill-defined.

  std::vector<double> post_distances {0.0, 1.0/2.0, 1.0};

  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
  }
}

TEST(ArgMaxTransformer, TransformPlane) {
  RMMatrixXf vertices(3, 3);
  vertices << 1.0, 2.0, 0.0,
              -1.0, 3.0, 0.0,
              1.0, 4.0, -1.0;

  // The argmax is the same for all points.
  RMMatrixXf truth_vertices(3, 1);
  truth_vertices << 1.0, 1.0, 1.0;
  RMMatrixXf truth_combinations(3, 3);
  truth_combinations.setIdentity();

  UPolytope polytope(&vertices, 2, {{0, 1, 2}});

  ArgMaxTransformer transformer;
  transformer.TransformUPolytope(&polytope);

  EXPECT_EQ(polytope.vertices(), truth_vertices);
  EXPECT_EQ(polytope.combinations(), truth_combinations);
}
