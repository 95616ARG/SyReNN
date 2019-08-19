#include <assert.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <unordered_map>
#include <utility>
#include "openblas/cblas.h"
#include "syrenn_server/segmented_line.h"

SegmentedLine::SegmentedLine(const RMVectorXf &start, const RMVectorXf &end)
    : endpoint_ratios_({0.0, 1.0}), interpolate_before_layer_({-1, -1}),
      preimage_start_(start), preimage_end_(end), preimage_delta_(end - start),
      point_dims_(start.size()), precomputed_until_(0) {
  ResetPrecomputed();
}

SegmentedLine::SegmentedLine(SegmentedLineStub *stub,
                             const RMVectorXf &start, const RMVectorXf &end)
    : endpoint_ratios_(std::move(stub->endpoint_ratios)),
      interpolate_before_layer_(std::move(stub->interpolate_before_layer)),
      applied_layers_(std::move(stub->applied_layers)),
      preimage_start_(start), preimage_end_(end), preimage_delta_(end - start),
      point_dims_(start.size()), precomputed_until_(0) {
  ResetPrecomputed();
}

void SegmentedLine::ResetPrecomputed() {
  precomputed_points_.resize(2, preimage_start_.cols());
  precomputed_points_.row(0) = preimage_start_;
  precomputed_points_.row(1) = preimage_end_;

  precomputed_until_ = 0;
}

SegmentedLine::~SegmentedLine() {
  // This is almost certainly not necessary, but we do it anyways to ensure
  // there's minimal chance of any unexpected memory overhead.
  endpoint_ratios_.clear();
  interpolate_before_layer_.clear();
  applied_layers_.clear();

  preimage_start_.resize(0);
  preimage_end_.resize(0);
  preimage_delta_.resize(0);
  precomputed_points_.resize(0, 0);
}

SegmentedLine SegmentedLine::Deserialize(
    const syrenn_server::SegmentedLine &line) {
  assert(line.endpoints_size() == 2);
  auto start = line.endpoints()[0];
  auto end = line.endpoints()[1];
  assert(start.preimage_ratio() == 0.0);
  assert(end.preimage_ratio() == 1.0);

  const RMVectorXf start_coordinates =
          Eigen::Map<const RMVectorXf>(
                          start.coordinates().data(),
                          start.coordinates_size());
  const RMVectorXf end_coordinates =
          Eigen::Map<const RMVectorXf>(
                          end.coordinates().data(),
                          end.coordinates_size());
  // NOTE: they should be in ascending distance
  return SegmentedLine(start_coordinates, end_coordinates);
}

syrenn_server::SegmentedLine SegmentedLine::Serialize(
    const std::vector<double> &endpoint_ratios, const RMMatrixXf &points) {
  auto serialized = syrenn_server::SegmentedLine();
  for (size_t i = 0; i < endpoint_ratios.size(); i++) {
    auto endpoint = serialized.add_endpoints();
    endpoint->set_preimage_ratio(endpoint_ratios[i]);

    if (i < static_cast<size_t>(points.rows())) {
      for (int j = 0; j < points.cols(); j++) {
        endpoint->add_coordinates(points(i, j));
      }
    }
  }

  return serialized;
}

const RMMatrixXf &SegmentedLine::points() const {
  return precomputed_points_;
}

size_t SegmentedLine::Size() const {
  return endpoint_ratios_.size();
}

size_t SegmentedLine::point_dims() const {
  return point_dims_;
}

size_t SegmentedLine::n_applied_layers() const {
  return applied_layers_.size();
}

double SegmentedLine::endpoint_ratio(size_t index) const {
  return endpoint_ratios_.at(index);
}

void SegmentedLine::PrecomputePoints() {
  for (size_t l = precomputed_until_; l < applied_layers_.size(); l++) {
    InterpolateBeforeLayer(static_cast<int>(l));
    auto layer = applied_layers_.at(l);
    // Do the computation
    // NOTE: This takes a lot of memory!
    layer(&precomputed_points_);
    precomputed_until_ = l + 1;
  }
}

void SegmentedLine::InsertEndpoints(std::vector<double> *endpoints,
                                    LayerComputeFunction layer,
                                    size_t point_dims) {
  // j holds the index into *endpoints, i holds the index into *this.
  size_t j = 0;
  for (size_t i = 1; i < Size(); i++) {
    // Insert all endpoints that should go between (i - 1) and i.
    double cutoff_ratio = endpoint_ratio(i);
    for (; j < endpoints->size() && endpoints->at(j) <= cutoff_ratio; j++) {
      if (endpoints->at(j) == cutoff_ratio) {
        // Just eat these.
        continue;
      }
      // Otherwise, insert them.
      endpoint_ratios_.insert(
          endpoint_ratios_.begin() + i, endpoints->at(j));
      interpolate_before_layer_.insert(
          interpolate_before_layer_.begin() + i, applied_layers_.size());
      i++;
    }
  }

  // Update the metadata of the line.
  applied_layers_.push_back(layer);
  point_dims_ = point_dims;
}

void SegmentedLine::InterpolateBeforeLayer(int layer) {
  size_t dims = precomputed_points_.cols();

  assert(precomputed_until_ == layer);

  std::queue<double> new_ratios;
  std::vector<size_t> precomputed_indices;
  size_t num_points = 0;
  for (size_t i = 0; i < Size(); i++) {
    if (interpolate_before_layer_.at(i) == layer) {
      new_ratios.push(endpoint_ratio(i));
      num_points++;
    }
    if (interpolate_before_layer_.at(i) < layer) {
      precomputed_indices.push_back(i);
      num_points++;
    }
  }

  assert(precomputed_indices.size() ==
         static_cast<size_t>(precomputed_points_.rows()));

  if (new_ratios.empty()) {
    return;
  }

  // NOTE: This will double the threshold!
  RMMatrixXf precomputed(num_points, dims);
  precomputed.setZero();

  size_t row = 0;
  RMVectorXf delta(dims);

  for (size_t i = 0; (i + 1) < precomputed_indices.size(); i++) {
    size_t from_index = precomputed_indices.at(i);
    size_t to_index = precomputed_indices.at(i + 1);

    precomputed.row(row) = precomputed_points_.row(i);
    row++;

    std::vector<float> midpoint_ratios;
    while (!new_ratios.empty() &&
            new_ratios.front() <= endpoint_ratio(to_index)) {
      midpoint_ratios.push_back(
              (new_ratios.front() - endpoint_ratio(from_index))
              / (endpoint_ratio(to_index) - endpoint_ratio(from_index)));
      new_ratios.pop();
    }
    delta = precomputed_points_.row(i + 1) - precomputed_points_.row(i);

    cblas_sger(CblasRowMajor, midpoint_ratios.size(),
               dims, 1.0, midpoint_ratios.data(), 1, delta.data(), 1,
               precomputed.data() + (row * dims), dims);

    precomputed
        .block(row, 0, midpoint_ratios.size(), dims)
        .rowwise() += precomputed_points_.row(i);

    row += midpoint_ratios.size();
  }

  // Add the end too
  precomputed.row(row) =
    precomputed_points_.row(precomputed_indices.size() - 1);

  // Swap it into the line instance
  // We've now precomputed right up to the next layer
  precomputed_points_.swap(precomputed);

  // Get rid of the old precomputed memory
  precomputed.resize(0, 0);
}

RMVectorXf SegmentedLine::InterpolatePreimage(double ratio) const {
  if (ratio == 0.0) {
    return preimage_start_;
  }
  if (ratio == 1.0) {
    return preimage_end_;
  }
  return preimage_start_ + (ratio * preimage_delta_);
}

// Removes things *AFTER* @end, not including @end.
void SegmentedLine::RemoveAfter(size_t end) {
  if (Size() == 0) {
    return;
  }
  end = std::min(end, Size() - 1);
  if (end == (Size() - 1)) {
    return;
  }

  size_t new_size = end + 1;

  // Update the preimages of the endpoint.
  preimage_end_ = InterpolatePreimage(endpoint_ratios_.at(end));
  preimage_delta_ = preimage_end_ - preimage_start_;

  // Remove all endpoints after end.
  endpoint_ratios_.resize(new_size);
  interpolate_before_layer_.resize(new_size);
  assert(Size() == new_size);

  // Next, we update the preimage-ratios of the endpoints we're keeping and
  // count up what the new size of precomputed_points_ should be.
  size_t num_precomputed = 0;
  for (size_t i = 0; i < Size(); i++) {
    if (IsPrecomputed(i)) {
      num_precomputed++;
    }
    endpoint_ratios_[i] = endpoint_ratio(i) / endpoint_ratio(Size() - 1);
  }

  // Finally, we update precomputed_points_. Note that the new preimage_end_
  // may not be precomputed yet; if so, we need to compute it.
  size_t new_precomputed = IsPrecomputed(Size() - 1) ?
                           num_precomputed : (num_precomputed + 1);
  precomputed_points_.conservativeResize(new_precomputed,
                                         precomputed_points_.cols());

  // Make sure the endpoint is precomputed, and update its
  // interpolate_before_layer_.
  interpolate_before_layer_.back() = -1;
  precomputed_points_.row(new_precomputed - 1) = ApplyAllLayers(preimage_end_);
}

RMVectorXf SegmentedLine::ApplyAllLayers(const RMVectorXf &preimage) const {
  RMMatrixXf postimage = preimage;
  for (int i = 0; i < precomputed_until_; i++) {
    applied_layers_.at(i)(&postimage);
  }
  assert(postimage.rows() == 1);
  return postimage.row(0);
}

// [start, end)
std::unique_ptr<SegmentedLineStub> SegmentedLine::ExtractStub(
        size_t start, size_t end) const {
  end = std::min(end, Size());

  std::vector<double> endpoint_ratios;
  std::vector<int> interpolate_before_layers;

  double subline_length = endpoint_ratio(end - 1) - endpoint_ratio(start);
  for (size_t i = start; i < end; i++) {
    if (i == start) {
      endpoint_ratios.push_back(0.0);
      interpolate_before_layers.push_back(-1);
    } else if ((i + 1) == end) {
      endpoint_ratios.push_back(1.0);
      interpolate_before_layers.push_back(-1);
    } else {
      double subline_ratio = (endpoint_ratio(i) - endpoint_ratio(start)) /
                             subline_length;
      endpoint_ratios.push_back(subline_ratio);
      interpolate_before_layers.push_back(interpolate_before_layer_.at(i));
    }
  }

  // The start/end points should always be interpolated at the beginning.
  interpolate_before_layers.front() = -1;
  interpolate_before_layers.back() = -1;

  std::unique_ptr<SegmentedLineStub> stub(new SegmentedLineStub(
        &endpoint_ratios, &interpolate_before_layers,
        applied_layers_));

  return stub;
}
