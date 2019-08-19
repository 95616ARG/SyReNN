#ifndef SYRENN_SYRENN_SERVER_SEGMENTED_LINE_H_
#define SYRENN_SYRENN_SERVER_SEGMENTED_LINE_H_

#include <assert.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/shared.h"
#include "eigen3/Eigen/Dense"

class SegmentedLine;
class SegmentedLineStub;

using SegmentedLineIterator = std::vector<double>::iterator;
using LayerComputeFunction = std::function<void(RMMatrixXf *)>;

// Represents SyReNN for a one-dimensional restriction domain of interest.
class SegmentedLine {
 public:
  // Constructor for "fresh" SegmentedLines, eg. those passed to the server.
  SegmentedLine(const RMVectorXf &start, const RMVectorXf &end);
  // Constructs a SegmentedLine from a "dormant" stub, and destroy the stub's
  // contents. NOTE: @stub is *NOT* freed here.
  SegmentedLine(SegmentedLineStub *stub,
                const RMVectorXf &start, const RMVectorXf &end);

  ~SegmentedLine();

  // Deserialization from protobuf.
  static SegmentedLine Deserialize(
      const syrenn_server::SegmentedLine &line);

  // Serialization to protobuf.
  // NOTE: We do this as a static method that doesn't actually rely on a
  // SegmentedLine at all because it simplifies the transform_big_line
  // implementation in server.cc. I have tried it the other way, and it's not
  // particularly pretty.
  static syrenn_server::SegmentedLine Serialize(
      const std::vector<double> &endpoint_ratios, const RMMatrixXf &points);

  // Inserts a set of proposed endpoints into SegmentedLine associated with a
  // particular layer. NOTE that this may be destructive to endpoints.
  void InsertEndpoints(std::vector<double> *endpoints,
                       LayerComputeFunction layer, size_t point_dims);

  // Helper method to return the preimage of ratio @ratio along the line.
  RMVectorXf InterpolatePreimage(double ratio) const;

  // Precomputes all points so that points() returns the post-image for all
  // points under all layers in applied_layers.
  void PrecomputePoints();

  // Returns post-images for all endpoints in the line under @applied_layers_
  // up to the last time PrecomputePoints was called.
  const RMMatrixXf &points() const;

  // Number of endpoints on the line.
  size_t Size() const;
  // Should match points().cols(), but does not force evaluation of points
  // (i.e., is constant-time).
  size_t point_dims() const;
  // Returns the number of layers that have transformed this function (i.e.
  // needed to compute post).
  size_t n_applied_layers() const;
  // Returns the preimage-ratio for the @index endpoint.
  double endpoint_ratio(size_t index) const;

  // Removes all endpoints *AFTER* (not including) index @end from the line.
  void RemoveAfter(size_t end);
  // Extracts a SegmentedLineStub from the subline with indices [@start, @end).
  std::unique_ptr<SegmentedLineStub> ExtractStub(size_t start,
                                                 size_t end) const;

 private:
  void ResetPrecomputed();

  // Adds points with interpolate_before_layer[index] = @layer to
  // precomputed_points_. NOTE: The line must already be precomputed_until_
  // @layer.
  void InterpolateBeforeLayer(int layer);

  RMVectorXf ApplyAllLayers(const RMVectorXf &preimage) const;

  bool IsPrecomputed(size_t index) {
    return interpolate_before_layer_.at(index) < precomputed_until_;
  }

  std::vector<double> endpoint_ratios_;
  std::vector<int> interpolate_before_layer_;

  std::vector<LayerComputeFunction> applied_layers_;

  RMVectorXf preimage_start_;
  RMVectorXf preimage_end_;
  RMVectorXf preimage_delta_;

  RMMatrixXf precomputed_points_;
  size_t point_dims_;
  int precomputed_until_;
};

struct SegmentedLineStub {
  SegmentedLineStub(std::vector<double> *endpoint_ratios,
                    std::vector<int> *interpolate_before_layer,
                    std::vector<LayerComputeFunction> applied_layers)
      : endpoint_ratios(std::move(*endpoint_ratios)),
        interpolate_before_layer(std::move(*interpolate_before_layer)),
        applied_layers(applied_layers) {}
  std::vector<double> endpoint_ratios;
  std::vector<int> interpolate_before_layer;

  std::vector<LayerComputeFunction> applied_layers;
};

#endif  // SYRENN_SYRENN_SERVER_SEGMENTED_LINE_H_
