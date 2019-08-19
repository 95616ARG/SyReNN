#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/relu_maxpool_transformer.h"
#include "syrenn_server/strided_window_data.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"

TEST(ReLUMaxPoolTransformer, Compute) {
  const size_t height = 4, width = 8, channels = 2;
  RMMatrixXf batch(channels, height * width);
  batch <<  // First channel.
           -1, -1, 1, 1, 1, 1, 1, 1,
           -1, -1, 1, 1, 1, 1, 1, 1,
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
  ReLUMaxPoolTransformer transformer(window_data);
  transformer.Compute(&batch);

  EXPECT_EQ(batch.rows(), 1);
  batch.resize(window_data.out_height() * window_data.out_width(), channels);
  batch = batch.transpose().eval();

  RMMatrixXf truth(2, 6);
  truth << 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
           2.0, 2.0, 2.0, 2.0, 2.0, 2.0;

  EXPECT_EQ(batch, truth);
}

TEST(ReLUMaxPoolTransformer, out_size) {
  const size_t height = 128, width = 256, channels = 5;
  const size_t window_height = 3, window_width = 4,
               stride_height = 2, stride_width = 5,
               pad_height = 1, pad_width = 3;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, channels, stride_height,
                                stride_width, pad_height, pad_width);
  ReLUMaxPoolTransformer transformer(window_data);
  EXPECT_EQ(transformer.out_size(height * width * channels),
            window_data.out_size());
}

TEST(ReLUMaxPoolTransformer, TransformLine) {
  const size_t height = 1, width = 4, channels = 1;
  const size_t window_height = 1, window_width = 2,
               stride_height = 1, stride_width = 2,
               pad_height = 0, pad_width = 1;
  StridedWindowData window_data(height, width, channels, window_height,
                                window_width, channels, stride_height,
                                stride_width, pad_height, pad_width);

  RMVectorXf startpoint(4);
  startpoint << -1, 2, 1, 3;
  RMVectorXf endpoint(4);
  endpoint << 1, 2, 5, 3;

  SegmentedLine line(startpoint, endpoint);

  ReLUMaxPoolTransformer transformer(window_data);
  transformer.TransformLine(&line);
  line.PrecomputePoints();

  EXPECT_EQ(4ul, line.Size());

  std::vector<double> post_distances{0.0, 1.0/4.0, 1.0/2.0, 1.0};
  for (size_t i = 0; i < post_distances.size(); i++) {
    EXPECT_EQ(post_distances[i], line.endpoint_ratio(i));
  }

  RMMatrixXf post_vertices(4, 3);
  post_vertices << 0, 2, 3,
                   0, 2, 3,
                   0, 3, 3,
                   1, 5, 3;
  EXPECT_EQ(line.points(), post_vertices);
}
