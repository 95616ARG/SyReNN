#include "gtest/gtest.h"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/conv2d_transformer.h"
#include "syrenn_server/strided_window_data.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(Conv2DTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto conv2d_data = serialized.mutable_conv2d_data();
  auto window_data = conv2d_data->mutable_window_data();
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
  auto deserialized = Conv2DTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(Conv2DTransformer, Compute) {
  const size_t points = 10,
               height = 4,
               width = 8,
               channels = 2,
               out_channels = 4;
  RMMatrixXf batch(points, height * width * channels);
  batch.setZero();

  const size_t window_height = 3, window_width = 4,
               stride_height = 2, stride_width = 5,
               pad_height = 1, pad_width = 3;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, out_channels, stride_height,
                                stride_width, pad_height, pad_width);

  RMMatrixXf filters(window_height * window_width, channels * out_channels);
  filters.setRandom();
  RMVectorXf biases(out_channels);
  biases.setZero();
  Conv2DTransformer transformer(filters, biases, window_data);

  transformer.Compute(&batch);

  EXPECT_EQ(batch.rows(), static_cast<int>(points));
  RMMatrixXf truth(points, window_data.out_size());
  truth.setZero();
  EXPECT_EQ(batch, truth);
}

TEST(Conv2DTransformer, out_size) {
  const size_t height = 128, width = 256, channels = 5, out_channels = 6;
  const size_t window_height = 3, window_width = 4,
               stride_height = 2, stride_width = 5,
               pad_height = 1, pad_width = 3;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, out_channels, stride_height,
                                stride_width, pad_height, pad_width);
  RMMatrixXf filters(window_height * window_width, channels * out_channels);
  RMVectorXf biases(out_channels);
  Conv2DTransformer transformer(filters, biases, window_data);
  EXPECT_EQ(transformer.out_size(height * width * channels),
            window_data.out_size());
}

// Transform methods are inherited from AffineTransformer, which we test
// separately.
