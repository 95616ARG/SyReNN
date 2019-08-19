#ifndef SYRENN_SYRENN_SERVER_MAXPOOL_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_MAXPOOL_TRANSFORMER_H_

#include <memory>
#include <string>
#include <vector>
#include "syrenn_proto/syrenn.grpc.pb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/transformer.h"
#include "syrenn_server/strided_window_data.h"
#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1
#include "tbb/concurrent_set.h"

// Transformer for MaxPool layers.
// NOTE: Currently, plane transforming is not supported for MaxPoolTransformer.
// NOTE: As with PyTorch, MaxPooling *DOES NOT* count the padded zeros as
// potential max-values.
class MaxPoolTransformer : public LayerTransformer {
 public:
  explicit MaxPoolTransformer(const StridedWindowData &window_data);

  static std::unique_ptr<LayerTransformer> Deserialize(
      const syrenn_server::Layer &layer);

  size_t out_size(size_t in_size) const override;

  // Computes ExactLine for a single window.
  virtual void process_window(
      // The windows in (H, W) format.
      const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &from_window,
      const Eigen::Map<RMMatrixXf, 0, Eigen::OuterStride<>> &to_window,
      // The preimage-ratios for each endpoint.
      double from_ratio, double to_ratio,
      // The set to place the endpoints in.
      tbb::concurrent_set<double> *endpoints) const;

  // Computes ExactLine between from_image and to_image.
  void process_images(
          // The images in (N, HWC) format.
          Eigen::Ref<const RMMatrixXf> from_image,
          Eigen::Ref<const RMMatrixXf> to_image,
          // The preimage-ratios for each endpoint.
          double from_ratio, double to_ratio,
          // The set to place the endpoints in.
          tbb::concurrent_set<double> *endpoints) const;

  void Compute(RMMatrixXf *inout) const;

  std::vector<double> ProposeLineEndpoints(
      const SegmentedLine &line) const override;

  std::string layer_type() const override {
    return "MaxPool";
  }

 protected:
  const StridedWindowData window_data_;
};

struct WindowPixel {
  WindowPixel() = default;
  WindowPixel(size_t h, size_t w) : h(h), w(w) {}

  bool operator==(const WindowPixel &other) const {
    return h == other.h && w == other.w;
  }

  size_t h;
  size_t w;
};

// https://stackoverflow.com/questions/17016175
namespace std {
template <>
struct hash<WindowPixel> {
  size_t operator()(const WindowPixel& k) const {
    size_t hash = 9;
    hash_combine(&hash, k.h, k.w);
    return hash;
  }
};
}

#endif  // SYRENN_SYRENN_SERVER_MAXPOOL_TRANSFORMER_H_
