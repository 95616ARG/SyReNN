#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/concat_transformer.h"
#include "syrenn_server/fullyconnected_transformer.h"
#include "syrenn_server/relu_transformer.h"
#include "syrenn_server/conv2d_transformer.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(ConcatTransformer, Deserialize) {

  syrenn_server::Layer serialized;
  auto concat_data = serialized.mutable_concat_data();
  concat_data->set_concat_along(
      syrenn_server::ConcatLayerData::CONCAT_ALONG_FLAT);

  auto fullyconnected_serialized = concat_data->add_layers();
  auto fullyconnected_data =
    fullyconnected_serialized->mutable_fullyconnected_data();
  fullyconnected_data->add_weights(1.0);
  fullyconnected_data->add_weights(1.0);
  fullyconnected_data->add_biases(0.0);

  LayerTransformer::RegisterDeserializer(
      &FullyConnectedTransformer::Deserialize);
  auto deserialized = ConcatTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
  EXPECT_EQ(deserialized->out_size(1), 1ul);
}

TEST(ConcatTransformer, ComputeAndTransformFlat) {
  const size_t n_points = 5, in_dims = 3, fullyconnected_out_dims = 1024;

  RMMatrixXf batch(n_points, in_dims);
  batch.setRandom();

  RMMatrixXf weights(in_dims, fullyconnected_out_dims);
  weights.setRandom();

  RMVectorXf biases(fullyconnected_out_dims);
  biases.setRandom();

  RMMatrixXf original_batch = batch;
  RMMatrixXf fullyconnected_truth = (batch * weights).rowwise() + biases;
  RMMatrixXf relu_truth = batch.array().max(0.0);

  std::vector<std::unique_ptr<LayerTransformer>> layers;
  layers.emplace_back(new FullyConnectedTransformer(weights, biases));
  layers.emplace_back(new ReLUTransformer());

  ConcatTransformer concat(&layers, ConcatAlong::FLAT);
  EXPECT_EQ(concat.out_size(in_dims), 1024ul + 3ul);

  concat.Compute(&batch);

  EXPECT_EQ(batch.rows(), 5);
  EXPECT_EQ(batch.cols(), 1024 + 3);
  EXPECT_EQ(batch.block(0, 0, 5, fullyconnected_out_dims),
                        fullyconnected_truth);
  EXPECT_EQ(batch.block(0, fullyconnected_out_dims, 5, in_dims),
                        relu_truth);

  batch = original_batch.array().abs();
  batch(0, 0) = -1.0;
  batch(1, 0) = +1.0;
  RMVectorXf start = batch.row(0);
  RMVectorXf end = batch.row(1);
  SegmentedLine line(start, end);
  concat.TransformLine(&line);
  EXPECT_EQ(line.Size(), 3ul);
  EXPECT_EQ(line.endpoint_ratio(0), 0.0);
  EXPECT_EQ(line.endpoint_ratio(1), 0.5);
  EXPECT_EQ(line.endpoint_ratio(2), 1.0);
  line.PrecomputePoints();

  RMMatrixXf truth = batch.block(0, 0, 3, in_dims);
  truth.row(2) = truth.row(1);
  truth.row(1) = 0.5 * (truth.row(0) + truth.row(1));
  concat.Compute(&truth);
  EXPECT_EQ(line.points().isApprox(truth), true);
}

TEST(ConcatTransformer, ComputeChannels) {
  const size_t points = 10,
               height = 1,
               width = 2,
               channels = 1,
               out_channels = 4;
  RMMatrixXf batch(points, height * width * channels);
  batch.setRandom();

  const size_t window_height = 1, window_width = 1,
               stride_height = 1, stride_width = 1,
               pad_height = 0, pad_width = 0;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, out_channels, stride_height,
                                stride_width, pad_height, pad_width);

  RMMatrixXf filters(window_height * window_width, channels * out_channels);
  filters.setZero();
  RMVectorXf biases(out_channels);
  biases.setConstant(1.0);
  RMVectorXf biases2(out_channels);
  biases2.setConstant(2.0);

  std::vector<std::unique_ptr<LayerTransformer>> layers;
  layers.emplace_back(new Conv2DTransformer(filters, biases, window_data));
  layers.emplace_back(new Conv2DTransformer(filters, biases2, window_data));

  ConcatTransformer concat(&layers, ConcatAlong::CHANNELS);
  EXPECT_EQ(concat.out_size(1ul), 8ul + 8ul);

  concat.Compute(&batch);

  EXPECT_EQ(batch.rows(), static_cast<int>(points));
  EXPECT_EQ(batch.cols(), 16);

  // Resize to (NHW, C)
  batch.resize(points * 2, 8);

  EXPECT_EQ((batch.block(0, 0, points * 2, 4).array() == 1.0).all(), true);
  EXPECT_EQ((batch.block(0, 4, points * 2, 4).array() == 2.0).all(), true);
}
