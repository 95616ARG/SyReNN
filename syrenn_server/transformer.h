#ifndef SYRENN_SYRENN_SERVER_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_TRANSFORMER_H_

#include <memory>
#include <unordered_set>
#include <vector>
#include <string>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "eigen3/Eigen/Dense"

// Represents a transformer that computes FHat-Extend for a single layer in a
// network.
class LayerTransformer {
 public:
  // A function signature for a "Deserializer" which takes a serialized layer
  // and converts it to an instance of a subclass of LayerTransformer.
  using TransformerDeserializer =
      std::unique_ptr<LayerTransformer> (*)(const syrenn_server::Layer&);
  // A list of registered "Deserializers" that will be used by the factory
  // Deserialize method below. Essentially, each item in this set corresponds
  // to a deserialization function for a particular layer type. The
  // Deserializer returns nullptr if it does not apply to the serialized layer.
  static std::unordered_set<TransformerDeserializer> Deserializers;
  // Should be called once for each layer type that will be supported by the
  // server.
  static void RegisterDeserializer(TransformerDeserializer deserializer);
  // Factory method that deserializes a Layer from the protobuf into a
  // LayerTransformer.
  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);

  // Given @line = fhat(g, X) for a one-dimensional X, computes and assigns
  // @line' = fhat(this_layer \circ g, X). Thresholder is a thresholder that
  // will be used to discard some vertices to save space at the expense of
  // accuracy (if precise transformation is desired, use NopThresholder).
  // This is effectively a front-end to TransformLineInPlace, which concerns
  // itself only with the transformation (not the thresholding).
  virtual void TransformLine(SegmentedLine *line) const;
  virtual std::vector<double>
      ProposeLineEndpoints(const SegmentedLine &line) const = 0;

  // Given @inout = fhat(g, X) for any bounded UPolytope X, computes and
  // assigns @inout' = fhat(this_layer \circ g, X).
  virtual void TransformUPolytope(UPolytope *inout) const {
    throw "Unimplemented";
  }
  // Computes and assigns @inout' = this_layer(@inout); i.e. performs a
  // concrete forward pass on a batch of points.
  virtual void Compute(RMMatrixXf *inout) const = 0;
  LayerComputeFunction ComputeFunction() const {
    return std::bind(&LayerTransformer::Compute, this, std::placeholders::_1);
  }

  // Returns the number of output dimensions of the range of this layer given
  // that the input point has dimensionality input_size.
  virtual size_t out_size(size_t input_size) const = 0;
  // Human-readable representation of the layer type. Used for
  // debugging/visualization purposes.
  virtual std::string layer_type() const { return "Unknown"; }
};

using TransformerDeserializer = LayerTransformer::TransformerDeserializer;

#endif  // SYRENN_SYRENN_SERVER_TRANSFORMER_H_
