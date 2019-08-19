#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/relu_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(ReLUTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto relu_data = serialized.mutable_relu_data();
  EXPECT_EQ(!relu_data, false);
  auto deserialized = ReLUTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(ReLUTransformer, Compute) {
  const size_t n_points = 1024, dimensions = 4096;

  RMMatrixXf batch(n_points, dimensions);
  batch.setRandom();
  RMMatrixXf truth = batch.array().max(0.0);

  ReLUTransformer transformer;
  transformer.Compute(&batch);
  EXPECT_EQ(batch, truth);
}

TEST(ReLUTransformer, out_size) {
  ReLUTransformer transformer;
  EXPECT_EQ(transformer.out_size(1234), 1234ul);
}

TEST(ReLUTransformer, Simple2DTest) {
  RMVectorXf startpoint(2);
  startpoint << -1.0, 3.0;
  RMVectorXf endpoint(2);
  endpoint << 3.0, -1.0;

  SegmentedLine line(startpoint, endpoint);

  ReLUTransformer transformer;
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(4ul, line.Size());

  RMMatrixXf post_points(4, 2);
  post_points << 0, 3,
                 0, 2,
                 2, 0,
                 3, 0;
  std::vector<double> post_distances {0.0, 1.0/4.0, 3.0/4.0, 1.0};

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
    EXPECT_EQ(post_points.row(i), line.points().row(i));
  }
}

bool is_polytope_certainly_subset(const UPolytope &super_container,
                                  const UPolytope &sub_container,
                                  size_t super_index,
                                  size_t sub_index) {
  for (size_t i = 0; i < sub_container.n_vertices(sub_index); i++) {
    bool found_vertex = false;
    for (size_t j = 0; j < super_container.n_vertices(super_index); j++) {
      if (super_container.vertex(super_index, j) ==
          sub_container.vertex(sub_index, i)) {
        found_vertex = true;
        break;
      }
    }
    if (!found_vertex) {
      return false;
    }
  }
  return true;
}

bool are_polytopes_certainly_equal(const UPolytope &polytope1,
                                   const UPolytope &polytope2,
                                   size_t index1,
                                   size_t index2) {
  return is_polytope_certainly_subset(polytope1, polytope2, index1, index2) &&
         is_polytope_certainly_subset(polytope2, polytope1, index2, index1);
}

void expect_certainly_equal(const UPolytope &polytope1,
                            const UPolytope &polytope2) {
  EXPECT_EQ(polytope1.n_polytopes(), polytope2.n_polytopes());
  for (size_t i = 0; i < polytope1.n_polytopes(); i++) {
    bool found = false;
    for (size_t j = 0; j < polytope2.n_polytopes(); j++) {
      if (are_polytopes_certainly_equal(polytope1, polytope2, i, j)) {
        found = true;
        break;
      }
    }
    EXPECT_EQ(found, true);
  }
}

TEST(ReLUTransformer, PlaneTransformer) {
  RMMatrixXf vertices(4, 2);
  vertices << 1.0, 1.0,
              -1.0, 1.0,
              -1.0, -1.0,
              1.0, -1.0;

  UPolytope polytope(&vertices, 2, {{0, 1, 2, 3}});

  ReLUTransformer transformer;
  transformer.TransformUPolytope(&polytope);

  RMMatrixXf truth_vertices(4, 2);
  truth_vertices << 1.0, 1.0,
                    0.0, 1.0,
                    0.0, 0.0,
                    1.0, 0.0;

  UPolytope truth_polytope(&truth_vertices, 2, {
    // (+, +) orthant
    {0, 1, 2, 3},
    // (-, +)
    {1, 2},
    // (+, -)
    {2, 3},
    // (-, -)
    {2, 2}
  });

  expect_certainly_equal(polytope, truth_polytope);
}
