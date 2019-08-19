#include "gtest/gtest.h"
#include "syrenn_server/segmented_line.h"

TEST(SegmentedLine, ConstructBlank) {
  RMVectorXf start(123);
  start.setRandom();
  RMVectorXf end(123);
  end.setRandom();

  SegmentedLine line(start, end);

  EXPECT_EQ(line.Size(), 2ul);
  EXPECT_EQ(line.point_dims(), 123ul);
  EXPECT_EQ(line.n_applied_layers(), 0ul);
  EXPECT_EQ(line.endpoint_ratio(0), 0.0);
  EXPECT_EQ(line.endpoint_ratio(1), 1.0);
  EXPECT_EQ(line.points().row(0), start);
  EXPECT_EQ(line.points().row(1), end);
}

void DoubleFunction(RMMatrixXf *inout) {
  (*inout) *= 2.0;
}

TEST(SegmentedLine, InsertEndpointsAndStubify) {
  RMVectorXf start(123);
  start.setRandom();
  RMVectorXf end(123);
  end.setRandom();

  SegmentedLine line(start, end);

  std::vector<double> endpoints{0.25, 0.5, 0.75};
  line.InsertEndpoints(&endpoints, &DoubleFunction, 123);

  EXPECT_EQ(line.Size(), 5ul);
  EXPECT_EQ(line.point_dims(), 123ul);
  EXPECT_EQ(line.n_applied_layers(), 1ul);
  EXPECT_EQ(line.endpoint_ratio(0), 0.0);
  EXPECT_EQ(line.endpoint_ratio(1), 0.25);
  EXPECT_EQ(line.endpoint_ratio(2), 0.5);
  EXPECT_EQ(line.endpoint_ratio(3), 0.75);
  EXPECT_EQ(line.endpoint_ratio(4), 1.0);
  EXPECT_EQ(line.points().rows(), 2);
  EXPECT_EQ(line.points().row(0), start);
  EXPECT_EQ(line.points().row(1), end);

  line.PrecomputePoints();
  EXPECT_EQ(line.points().rows(), 5);
  EXPECT_EQ(line.points().row(0).isApprox(2.0 * start), true);
  EXPECT_EQ(line.points().row(1).isApprox(
        2.0 * ((0.75 * start) + (0.25 * end))), true);
  EXPECT_EQ(line.points().row(2).isApprox(
        2.0 * ((0.5 * start) + (0.5 * end))), true);
  EXPECT_EQ(line.points().row(3).isApprox(
        2.0 * ((0.25 * start) + (0.75 * end))), true);
  EXPECT_EQ(line.points().row(4).isApprox(2.0 * end), true);

  auto stub = line.ExtractStub(1, 4);
  EXPECT_EQ(line.Size(), 5ul);
  EXPECT_EQ(stub->endpoint_ratios.size(), 3ul);
  EXPECT_EQ(stub->endpoint_ratios[0], 0.0);
  EXPECT_EQ(stub->endpoint_ratios[1], 0.5);
  EXPECT_EQ(stub->endpoint_ratios[2], 1.0);
  EXPECT_EQ(stub->interpolate_before_layer.size(), 3ul);
  EXPECT_EQ(stub->interpolate_before_layer[0], -1);
  EXPECT_EQ(stub->interpolate_before_layer[1], 0);
  EXPECT_EQ(stub->interpolate_before_layer[2], -1);
  EXPECT_EQ(stub->applied_layers.size(), 1ul);
  EXPECT_EQ(stub->applied_layers[0] == nullptr, false);

  line.RemoveAfter(2);
  EXPECT_EQ(line.Size(), 3ul);
  EXPECT_EQ(line.point_dims(), 123ul);
  EXPECT_EQ(line.n_applied_layers(), 1ul);
  EXPECT_EQ(line.endpoint_ratio(0), 0.0);
  EXPECT_EQ(line.endpoint_ratio(1), 0.25 / 0.5);
  EXPECT_EQ(line.endpoint_ratio(2), 1.0);
  EXPECT_EQ(line.points().rows(), 3);
  EXPECT_EQ(line.points().row(0).isApprox(2.0 * start), true);
  EXPECT_EQ(line.points().row(1).isApprox(
        2.0 * ((0.75 * start) + (0.25 * end))), true);
  EXPECT_EQ(line.points().row(2).isApprox(
        2.0 * ((0.5 * start) + (0.5 * end))), true);

  SegmentedLine from_stub(stub.get(),
                          ((0.75 * start) + (0.25 * end)).eval(),
                          ((0.25 * start) + (0.75 * end)).eval());
  EXPECT_EQ(from_stub.Size(), 3ul);
  EXPECT_EQ(from_stub.point_dims(), 123ul);
  EXPECT_EQ(from_stub.n_applied_layers(), 1ul);
  EXPECT_EQ(from_stub.endpoint_ratio(0), 0.0);
  EXPECT_EQ(from_stub.endpoint_ratio(1), 0.5);
  EXPECT_EQ(from_stub.endpoint_ratio(2), 1.0);
  EXPECT_EQ(from_stub.points().rows(), 2);
  EXPECT_EQ(from_stub.points().row(0).isApprox(
            (0.75 * start) + (0.25 * end)), true);
  EXPECT_EQ(from_stub.points().row(1).isApprox(
            (0.25 * start) + (0.75 * end)), true);

  from_stub.PrecomputePoints();
  EXPECT_EQ(from_stub.Size(), 3ul);
  EXPECT_EQ(from_stub.point_dims(), 123ul);
  EXPECT_EQ(from_stub.n_applied_layers(), 1ul);
  EXPECT_EQ(from_stub.endpoint_ratio(0), 0.0);
  EXPECT_EQ(from_stub.endpoint_ratio(1), 0.5);
  EXPECT_EQ(from_stub.endpoint_ratio(2), 1.0);
  EXPECT_EQ(from_stub.points().rows(), 3);
  EXPECT_EQ(from_stub.points().row(0).isApprox(
            (2.0 * ((0.75 * start) + (0.25 * end)))), true);
  EXPECT_EQ(from_stub.points().row(1).isApprox(
            (2.0 * ((0.5 * start) + (0.5 * end)))), true);
  EXPECT_EQ(from_stub.points().row(2).isApprox(
            (2.0 * ((0.25 * start) + (0.75 * end)))), true);

}
