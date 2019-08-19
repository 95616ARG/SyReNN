#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/averagepool_transformer.h"
#include "syrenn_server/strided_window_data.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(AveragePoolTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto averagepool_data = serialized.mutable_averagepool_data();
  auto window_data = averagepool_data->mutable_window_data();
  window_data->set_in_height(128);
  window_data->set_in_width(128);
  window_data->set_in_channels(3);
  window_data->set_window_height(4);
  window_data->set_window_width(4);
  window_data->set_out_channels(3);
  window_data->set_stride_height(4);
  window_data->set_stride_width(4);
  window_data->set_pad_height(1);
  window_data->set_pad_width(1);
  auto deserialized = AveragePoolTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(AveragePoolTransformer, Compute) {
  const size_t height = 4, width = 8, channels = 2;
  RMMatrixXf batch(channels, height * width);
  batch <<  // First channel.
           1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1,
           // Second channel.
           2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2;
  batch = batch.transpose().eval();
  batch.resize(1, height * width * channels);

  const size_t window_height = 3, window_width = 4,
               stride_height = 2, stride_width = 5,
               pad_height = 1, pad_width = 3;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, channels, stride_height,
                                stride_width, pad_height, pad_width);
  AveragePoolTransformer transformer(window_data);
  transformer.Compute(&batch);

  EXPECT_EQ(batch.rows(), 1);
  batch.resize(window_data.out_height() * window_data.out_width(), channels);
  batch = batch.transpose().eval();

  RMMatrixXf regress_truth(2, 6);
  regress_truth << 0.166667, 0.666667, 0.166667, 0.25, 1, 0.25,
                   0.333333, 1.33333, 0.333333, 0.5, 2, 0.5;

  EXPECT_EQ(batch.isApprox(regress_truth), true);
}

TEST(AveragePoolTransformer, out_size) {
  const size_t height = 128, width = 256, channels = 5;
  const size_t window_height = 3, window_width = 4,
               stride_height = 2, stride_width = 5,
               pad_height = 1, pad_width = 3;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, channels, stride_height,
                                stride_width, pad_height, pad_width);
  AveragePoolTransformer transformer(window_data);
  EXPECT_EQ(transformer.out_size(height * width * channels),
            window_data.out_size());
}

// Transform methods are inherited from AffineTransformer, which we test
// separately.
