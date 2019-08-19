#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/fullyconnected_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(FullyConnectedTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto fullyconnected_data = serialized.mutable_fullyconnected_data();
  fullyconnected_data->add_weights(1.0);
  fullyconnected_data->add_weights(1.0);
  fullyconnected_data->add_biases(0.0);
  EXPECT_EQ(!fullyconnected_data, false);
  auto deserialized = FullyConnectedTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(FullyConnectedTransformer, Compute) {
  const size_t n_points = 1024, in_dims = 4096, out_dims = 1024;

  RMMatrixXf batch(n_points, in_dims);
  batch.setRandom();

  RMMatrixXf weights(in_dims, out_dims);
  weights.setRandom();

  RMVectorXf biases(1, out_dims);
  biases.setRandom();

  RMMatrixXf truth = (batch * weights).rowwise() + biases;

  FullyConnectedTransformer transformer(weights, biases);
  transformer.Compute(&batch);
  EXPECT_EQ(batch, truth);
}

TEST(FullyConnectedTransformer, out_size) {
  const size_t in_dims = 4096, out_dims = 1024;

  RMMatrixXf weights(in_dims, out_dims);
  RMVectorXf biases(1, out_dims);

  FullyConnectedTransformer transformer(weights, biases);
  EXPECT_EQ(transformer.out_size(in_dims), out_dims);
}

// Transform methods are inherited from AffineTransformer, which we test
// separately.
