#include <utility>
#include "syrenn_server/upolytope.h"

UPolytope::UPolytope(RMMatrixXf *vertices, size_t subspace_dimensions,
                     std::vector<std::vector<size_t>> polytopes)
    : vertices_(0, 0), combinations_(vertices->rows(), vertices->rows()),
      subspace_dimensions_(subspace_dimensions),
      polytopes_(polytopes.begin(), polytopes.end()) {
  vertices_.swap(*vertices);
  combinations_.setIdentity();
}

UPolytope UPolytope::Deserialize(
    const syrenn_server::UPolytope &upolytope) {
  // TODO(masotoud): assume that combinations is empty.
  int n_vertices = 0;
  for (int i = 0; i < upolytope.polytopes_size(); i++) {
    auto &polytope = upolytope.polytopes()[i];
    n_vertices += polytope.num_vertices();
  }
  RMMatrixXf vertices(n_vertices, upolytope.space_dimensions());
  std::vector<std::vector<size_t>> polytopes(upolytope.polytopes_size());
  int vertices_index = 0;
  for (int i = 0; i < upolytope.polytopes_size(); i++) {
    auto &polytope = upolytope.polytopes()[i];
    vertices.block(vertices_index, 0,
                   polytope.num_vertices(),
                   upolytope.space_dimensions()) =
        Eigen::Map<const RMMatrixXf>(polytope.vertices().data(),
                                     polytope.num_vertices(),
                                     upolytope.space_dimensions());
    for (size_t j = 0; j < polytope.num_vertices(); j++) {
      polytopes[i].push_back(vertices_index);
      vertices_index++;
    }
  }
  return UPolytope(&vertices, upolytope.subspace_dimensions(), polytopes);
}

syrenn_server::UPolytope UPolytope::Serialize() const {
  auto serialized = syrenn_server::UPolytope();
  serialized.set_space_dimensions(vertices_.cols());
  serialized.set_subspace_dimensions(subspace_dimensions_);
  for (size_t polytope = 0; polytope < polytopes_.size(); polytope++) {
    auto serialized_polytope = serialized.add_polytopes();
    serialized_polytope->set_num_vertices(n_vertices(polytope));
    for (size_t v = 0; v < n_vertices(polytope); v++) {
      for (size_t dim = 0; dim < space_dimensions(); dim++) {
        serialized_polytope->add_vertices(vertex(polytope, v)(dim));
      }
      for (int comb_dim = 0; comb_dim < combinations_.cols(); comb_dim++) {
        serialized_polytope->add_combinations(
            combination(polytope, v)(comb_dim));
      }
    }
  }
  return serialized;
}

void UPolytope::FlushPending() {
  if (!pending_.empty()) {
    size_t n_old = vertices_.rows();
    size_t n_pending = pending_.size();
    vertices_.conservativeResize(n_old + n_pending, vertices_.cols());
    combinations_.conservativeResize(n_old + n_pending, combinations_.cols());
    for (size_t i = 0; i < n_pending; i++) {
      vertices_.row(n_old + i) = pending_[i].vertex;
      combinations_.row(n_old + i) = pending_[i].combination;
    }
    pending_.clear();
  }
}

RMMatrixXf &UPolytope::vertices() {
  FlushPending();
  return vertices_;
}

RMMatrixXf &UPolytope::combinations() {
  FlushPending();
  return combinations_;
}

std::vector<size_t> &UPolytope::vertex_indices(size_t polytope) {
  return polytopes_[polytope];
}

bool UPolytope::is_counter_clockwise() const {
  return subspace_dimensions_ == 2;
}

size_t UPolytope::space_dimensions() const {
  return vertices_.cols();
}

size_t UPolytope::n_polytopes() const {
  return polytopes_.size();
}

size_t UPolytope::n_vertices(size_t polytope) const {
  return polytopes_[polytope].size();
}

size_t UPolytope::vertex_index(size_t polytope, size_t vertex) const {
  return polytopes_[polytope][vertex];
}

Eigen::Ref<const RMVectorXf> UPolytope::vertex(size_t raw_index) const {
  if (raw_index >= static_cast<size_t>(vertices_.rows())) {
    return pending_[raw_index - vertices_.rows()].vertex;
  }
  return vertices_.row(raw_index);
}

Eigen::Ref<const RMVectorXf> UPolytope::vertex(size_t polytope,
                                               size_t vertex) const {
  return this->vertex(vertex_index(polytope, vertex));
}

Eigen::Ref<const RMVectorXf> UPolytope::combination(size_t raw_index) const {
  if (raw_index >= static_cast<size_t>(combinations_.rows())) {
    return pending_[raw_index - combinations_.rows()].combination;
  }
  return combinations_.row(raw_index);
}

Eigen::Ref<const RMVectorXf> UPolytope::combination(size_t polytope,
                                                    size_t vertex) const {
  return combination(vertex_index(polytope, vertex));
}

size_t UPolytope::AppendVertex(RMVectorXf *vertex, RMVectorXf *combination) {
  auto iterator = pending_.emplace_back(vertex, combination);
  // NOTE: This is just a hacky way to get my_index from the iterator. We
  // assume pending_.size() - 1 is a good first guess because:
  // (1) We know that index has to be less than pending_.size(), since we only
  // ever push to the back of the vector.
  // (2) If nothing else has been inserted, then it will be exactly
  // pending_.size() - 1.
  size_t index = pending_.size() - 1;
  auto iterator2 = pending_.begin() + index;
  for (; iterator2 != iterator; index--, iterator2--) {}
  return vertices_.rows() + index;
}

size_t UPolytope::AppendPolytope(std::vector<size_t> *vertex_indices) {
  auto iterator = polytopes_.emplace_back(std::move(*vertex_indices));
  // See AppendVertex above about this hack.
  size_t index = polytopes_.size() - 1;
  auto iterator2 = polytopes_.begin() + index;
  for (; iterator2 != iterator; index--, iterator2--) {}
  return index;
}
