#include "gtest/gtest.h"
#include "syrenn_server/upolytope.h"

void EXPECT_VECTORS_EQ(const std::vector<size_t> &v1,
                       const std::vector<size_t> &v2) {
  EXPECT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); i++) {
    EXPECT_EQ(v1[i], v2[i]);
  }
}

TEST(UPolytope, ConstructAndUse) {
  RMMatrixXf vertices(6, 3);
  vertices.setRandom();
  RMMatrixXf original_vertices = vertices;
  RMMatrixXf original_combinations(6, 6);
  original_combinations.setIdentity();

  UPolytope upolytope(&vertices, 2, {{0, 2, 3}, {1, 2, 4}, {3, 4, 5}});
  EXPECT_EQ(upolytope.vertices(), original_vertices);
  EXPECT_EQ(upolytope.combinations(), original_combinations);
  EXPECT_VECTORS_EQ(upolytope.vertex_indices(0), {0, 2, 3});
  EXPECT_VECTORS_EQ(upolytope.vertex_indices(1), {1, 2, 4});
  EXPECT_VECTORS_EQ(upolytope.vertex_indices(2), {3, 4, 5});
  EXPECT_EQ(upolytope.is_counter_clockwise(), true);
  EXPECT_EQ(upolytope.space_dimensions(), 3ul);
  EXPECT_EQ(upolytope.n_polytopes(), 3ul);
  EXPECT_EQ(upolytope.n_vertices(0), 3ul);
  EXPECT_EQ(upolytope.n_vertices(1), 3ul);
  EXPECT_EQ(upolytope.n_vertices(2), 3ul);
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(upolytope.vertex(i), original_vertices.row(i));
    EXPECT_EQ(upolytope.combination(i), original_combinations.row(i));
  }
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(upolytope.vertex_index(i, j), upolytope.vertex_indices(i)[j]);
      EXPECT_EQ(upolytope.vertex(i, j),
                upolytope.vertex(upolytope.vertex_indices(i)[j]));
      EXPECT_EQ(upolytope.combination(i, j),
                upolytope.combination(upolytope.vertex_indices(i)[j]));
    }
  }

  RMVectorXf new_vertex(3);
  new_vertex << 1, 2, 3;
  RMVectorXf new_vertex_ = new_vertex;
  RMVectorXf new_combination(6);
  new_combination << 0.75, 0.25, 0, 0, 0, 0;
  RMVectorXf new_combination_ = new_combination;
  size_t index = upolytope.AppendVertex(&new_vertex, &new_combination);
  EXPECT_EQ(upolytope.vertex(index), new_vertex_);
  EXPECT_EQ(upolytope.combination(index), new_combination_);
  EXPECT_EQ(upolytope.vertices().row(6), new_vertex_);
  EXPECT_EQ(upolytope.combinations().row(6), new_combination_);

  std::vector<size_t> indices{1, 2, 3};
  index = upolytope.AppendPolytope(&indices);
  EXPECT_EQ(upolytope.n_polytopes(), 4ul);
  EXPECT_VECTORS_EQ(upolytope.vertex_indices(index), {1, 2, 3});
}
