#include "gtest/gtest.h"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/strided_window_data.h"

void EXPECT_VECTORS_EQ(const std::vector<long int> &v1,
                       const std::vector<int> &v2) {
  EXPECT_EQ(v1.size(), v2.size());
  for (size_t i = 0; i < v1.size(); i++) {
    EXPECT_EQ(v1[i], v2[i]);
  }
}

TEST(StridedWindowData, ConstructAndQuery) {
  StridedWindowData window_data(128, 256, 3, 4, 5, 6, 7, 8, 9, 10);

  EXPECT_EQ(window_data.in_height(), 128ul);
  EXPECT_EQ(window_data.in_width(), 256ul);
  EXPECT_EQ(window_data.in_channels(), 3ul);

  EXPECT_EQ(window_data.window_height(), 4ul);
  EXPECT_EQ(window_data.window_width(), 5ul);
  EXPECT_EQ(window_data.out_channels(), 6ul);

  EXPECT_EQ(window_data.stride_height(), 7ul);
  EXPECT_EQ(window_data.stride_width(), 8ul);

  EXPECT_EQ(window_data.pad_height(), 9ul);
  EXPECT_EQ(window_data.pad_width(), 10ul);

  EXPECT_EQ(window_data.out_height(), 21ul);
  EXPECT_EQ(window_data.out_width(), 34ul);
  EXPECT_EQ(window_data.out_size(), 21ul * 34ul * 6ul);

  EXPECT_EQ(window_data.padded_in_height(), 128ul + (2ul * 9ul));
  EXPECT_EQ(window_data.padded_in_width(), 256ul + (2ul * 10ul));

  std::vector<int> input_dims{11, 3, 128, 256};
  EXPECT_VECTORS_EQ(window_data.mkl_input_dims(11), input_dims);

  std::vector<int> filter_dims{6, 3, 4, 5};
  EXPECT_VECTORS_EQ(window_data.mkl_filter_dims(), filter_dims);

  std::vector<int> window_dims{4, 5};
  EXPECT_VECTORS_EQ(window_data.mkl_window_dims(), window_dims);

  std::vector<int> bias_dims{6};
  EXPECT_VECTORS_EQ(window_data.mkl_bias_dims(), bias_dims);

  std::vector<int> output_dims{11, 6, 21, 34};
  EXPECT_VECTORS_EQ(window_data.mkl_output_dims(11), output_dims);

  std::vector<int> stride_dims{7, 8};
  EXPECT_VECTORS_EQ(window_data.mkl_stride_dims(), stride_dims);

  std::vector<int> pad_dims{9, 10};
  EXPECT_VECTORS_EQ(window_data.mkl_pad_dims(), pad_dims);
}

TEST(StridedWindowData, Deserialize) {
  syrenn_server::StridedWindowData serialized;
  serialized.set_in_height(128);
  serialized.set_in_width(256);
  serialized.set_in_channels(3);

  serialized.set_window_height(4);
  serialized.set_window_width(5);
  serialized.set_out_channels(6);

  serialized.set_stride_height(7);
  serialized.set_stride_width(8);

  serialized.set_pad_height(9);
  serialized.set_pad_width(10);

  auto window_data = StridedWindowData::Deserialize(serialized);

  EXPECT_EQ(window_data.in_height(), 128ul);
  EXPECT_EQ(window_data.in_width(), 256ul);
  EXPECT_EQ(window_data.in_channels(), 3ul);

  EXPECT_EQ(window_data.window_height(), 4ul);
  EXPECT_EQ(window_data.window_width(), 5ul);
  EXPECT_EQ(window_data.out_channels(), 6ul);

  EXPECT_EQ(window_data.stride_height(), 7ul);
  EXPECT_EQ(window_data.stride_width(), 8ul);

  EXPECT_EQ(window_data.pad_height(), 9ul);
  EXPECT_EQ(window_data.pad_width(), 10ul);
}
