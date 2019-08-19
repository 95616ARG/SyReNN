#include <memory>
#include <unordered_set>
#include "syrenn_server/transformer.h"
#include "syrenn_proto/syrenn.grpc.pb.h"

std::unordered_set<TransformerDeserializer> LayerTransformer::Deserializers;

void LayerTransformer::RegisterDeserializer(
    TransformerDeserializer deserializer) {
  Deserializers.insert(deserializer);
}

std::unique_ptr<LayerTransformer> LayerTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  for (auto deserializer : Deserializers) {
    std::unique_ptr<LayerTransformer> transformer = deserializer(layer);
    if (transformer) {
      return transformer;
    }
  }
  return nullptr;
}

void LayerTransformer::TransformLine(SegmentedLine *line) const {
  line->PrecomputePoints();
  std::vector<double> endpoints = ProposeLineEndpoints(*line);
  line->InsertEndpoints(&endpoints, ComputeFunction(),
                        out_size(line->point_dims()));
}
