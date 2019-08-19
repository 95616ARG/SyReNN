#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_server/hard_tanh_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(HardTanhTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto hard_tanh_data = serialized.mutable_hard_tanh_data();
  EXPECT_EQ(!hard_tanh_data, false);
  auto deserialized = HardTanhTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(HardTanhTransformer, Compute) {
  const size_t n_points = 1024, dimensions = 4096;

  RMMatrixXf batch(n_points, dimensions);
  batch.setRandom();
  RMMatrixXf truth = batch.array().max(-1.0).min(1.0);

  HardTanhTransformer transformer;
  transformer.Compute(&batch);

  EXPECT_EQ(batch, truth);
}

TEST(HardTanhTransformer, out_size) {
  HardTanhTransformer transformer;
  EXPECT_EQ(transformer.out_size(1024), 1024ul);
}

TEST(HardTanhTransformer, TransformLine) {
  RMVectorXf startpoint(2);
  startpoint << -2.0, 0.5;
  RMVectorXf endpoint(2);
  endpoint << 2.0, -0.5;

  SegmentedLine line(startpoint, endpoint);

  HardTanhTransformer transformer;
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(4ul, line.Size());

  std::vector<double> post_distances{0.0, 1.0/4.0, 3.0/4.0, 1.0};
  for (size_t i = 0; i < post_distances.size(); i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
  }

  RMMatrixXf post_vertices(4, 2);
  post_vertices << -1.0, 0.5,
                   -1.0, 0.25,
                    1.0, -0.25,
                    1.0, -0.5;
  EXPECT_EQ(line.points(), post_vertices);
}

TEST(HardTanhTransformer, TransformPlane) {
  RMMatrixXf vertices(3, 2);
  vertices << 1.0, 0.0,
              -1.0, 0.5,
              1.0, -1.0;

  RMMatrixXf truth_vertices = vertices;
  RMMatrixXf truth_combinations(3, 3);
  truth_combinations.setIdentity();

  UPolytope polytope(&vertices, 2, {{0, 1, 2}});

  HardTanhTransformer transformer;
  transformer.TransformUPolytope(&polytope);

  EXPECT_EQ(polytope.vertices(), truth_vertices);
  EXPECT_EQ(polytope.combinations(), truth_combinations);
}
