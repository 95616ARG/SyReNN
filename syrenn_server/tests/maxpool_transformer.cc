#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/maxpool_transformer.h"
#include "syrenn_server/strided_window_data.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(MaxPoolTransformer, Deserialize) {
  syrenn_server::Layer serialized;
  auto maxpool_data = serialized.mutable_maxpool_data();
  auto window_data = maxpool_data->mutable_window_data();
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
  auto deserialized = MaxPoolTransformer::Deserialize(serialized);
  EXPECT_EQ(!deserialized, false);
}

TEST(MaxPoolTransformer, Compute) {
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
  MaxPoolTransformer transformer(window_data);
  transformer.Compute(&batch);

  EXPECT_EQ(batch.rows(), 1);
  batch.resize(window_data.out_height() * window_data.out_width(), channels);
  batch = batch.transpose().eval();

  RMMatrixXf truth(2, 6);
  truth << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           2.0, 2.0, 2.0, 2.0, 2.0, 2.0;

  EXPECT_EQ(batch.isApprox(truth), true);
}

TEST(MaxPoolTransformer, out_size) {
  const size_t height = 128, width = 256, channels = 5;
  const size_t window_height = 3, window_width = 4,
               stride_height = 2, stride_width = 5,
               pad_height = 1, pad_width = 3;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, channels, stride_height,
                                stride_width, pad_height, pad_width);
  MaxPoolTransformer transformer(window_data);
  EXPECT_EQ(transformer.out_size(height * width * channels),
            window_data.out_size());
}

TEST(MaxPoolTransformer, TransformLine) {
  const size_t height = 1, width = 4, channels = 1;
  const size_t window_height = 1, window_width = 2,
               stride_height = 1, stride_width = 2,
               pad_height = 0, pad_width = 1;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, channels, stride_height,
                                stride_width, pad_height, pad_width);

  RMVectorXf startpoint(4);
  startpoint << 1, 2, 1, 3;
  RMVectorXf endpoint(4);
  endpoint << 1, 2, 5, 3;

  SegmentedLine line(startpoint, endpoint);

  MaxPoolTransformer transformer(window_data);
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(3ul, line.Size());

  std::vector<double> post_distances{0.0, 1.0/4.0, 1.0};
  for (size_t i = 0; i < post_distances.size(); i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
  }

  RMMatrixXf post_vertices(3, 3);
  post_vertices << 1, 2, 3,
                   1, 2, 3,
                   1, 5, 3;
  EXPECT_EQ(line.points(), post_vertices);
}
