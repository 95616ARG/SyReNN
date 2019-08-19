#include <iostream>
#include <set>
#include <utility>
#include "eigen3/Eigen/Dense"
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/concat_transformer.h"
#include "syrenn_server/conv2d_transformer.h"

ConcatTransformer::ConcatTransformer(
        std::vector<std::unique_ptr<LayerTransformer>> *input_transformers,
        const ConcatAlong &concat_along)
    : input_transformers_(std::move(*input_transformers)),
      concat_along_(concat_along) {}

size_t ConcatTransformer::out_size(size_t in_size) const {
  size_t out_size = 0;
  for (const auto &transformer : input_transformers_) {
    out_size += transformer->out_size(in_size);
  }
  return out_size;
}

std::unique_ptr<LayerTransformer> ConcatTransformer::Deserialize(
      const syrenn_server::Layer &layer) {
  if (!layer.has_concat_data()) {
    return nullptr;
  }
  std::vector<std::unique_ptr<LayerTransformer>> layers;
  for (auto &layer_data : layer.concat_data().layers()) {
    layers.push_back(LayerTransformer::Deserialize(layer_data));
  }
  ConcatAlong concat_along;
  switch (layer.concat_data().concat_along()) {
    case syrenn_server::ConcatLayerData::CONCAT_ALONG_CHANNELS:
      concat_along = ConcatAlong::CHANNELS;
      break;
    case syrenn_server::ConcatLayerData::CONCAT_ALONG_FLAT:
      concat_along = ConcatAlong::FLAT;
      break;
    default:
      throw "Unsupported concat_along type.";
      break;
  }
  return std::unique_ptr<LayerTransformer>(
            new ConcatTransformer(&layers, concat_along));
}

std::vector<double> ConcatTransformer::ProposeLineEndpoints(
    const SegmentedLine &line) const {
  std::vector<double> merged;
  size_t n_merged = 0;
  for (auto &transformer : input_transformers_) {
    std::vector<double> unmerged = transformer->ProposeLineEndpoints(line);

    merged.insert(std::end(merged), std::begin(unmerged), std::end(unmerged));

    std::inplace_merge(std::begin(merged),
                       std::begin(merged) + n_merged,
                       std::end(merged));

    n_merged = merged.size();
  }
  return merged;
}

void ConcatTransformer::Compute(RMMatrixXf *inout) const {
  size_t batch = inout->rows();

  std::vector<RMMatrixXf> transformed;
  size_t concat_dim_size = 0;
  for (const auto &transformer : input_transformers_) {
    RMMatrixXf computed = *inout;
    transformer->Compute(&computed);

    if (concat_along_ == ConcatAlong::CHANNELS) {
      auto channels =
        dynamic_cast<Conv2DTransformer*>(transformer.get())->out_channels();
      // transformed is (N, HWC), we want (NHW, C)
      computed.resize(computed.size() / channels, channels);
      concat_dim_size += channels;
    } else if (concat_along_ == ConcatAlong::FLAT) {
      concat_dim_size += computed.cols();
    } else {
      throw "Unsupported concat_along type.";
    }
    transformed.push_back(computed);
  }

  // NOTE(masotoud): If memory becomes a huge bottleneck, we can use one of the
  // matrices in transformed as storage space then swap instead of doubling the
  // memory needed here.
  inout->resize(transformed.front().rows(), concat_dim_size);
  unsigned int start_col = 0;
  for (RMMatrixXf &transformed_matrix : transformed) {
    inout->block(0, start_col, inout->rows(), transformed_matrix.cols())
        = transformed_matrix;
    start_col += transformed_matrix.cols();
    transformed_matrix.resize(0, 0);
  }
  inout->resize(batch, inout->size() / batch);
}
