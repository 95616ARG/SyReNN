#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/normalize_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(NormalizeTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto normalize_data = serialized.mutable_normalize_data();
  normalize_data->add_means(-1.0);
  normalize_data->add_means(+1.0);
  normalize_data->add_standard_deviations(+2.0);
  normalize_data->add_standard_deviations(-2.0);
  auto deserialized = NormalizeTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(NormalizeTransformer, Compute) {
  const size_t n_points = 1024, dims = 4096;

  RMMatrixXf batch(n_points, dims);
  batch.setRandom();

  RMVectorXf means(dims);
  means.setRandom();

  RMVectorXf standard_deviations(dims);
  standard_deviations.setRandom();

  RMMatrixXf truth = (batch.array().rowwise() - means.array()).rowwise() /
                     standard_deviations.array();

  NormalizeTransformer transformer(means, standard_deviations);
  transformer.Compute(&batch);
  EXPECT_EQ(batch, truth);
}

TEST(NormalizeTransformer, out_size) {
  const size_t dims = 1024;

  RMVectorXf means(dims);
  RMVectorXf standard_deviations(dims);

  NormalizeTransformer transformer(means, standard_deviations);
  EXPECT_EQ(transformer.out_size(dims), dims);
}

// Transform methods are inherited from AffineTransformer, which we test
// separately.
