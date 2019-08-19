#include "syrenn_server/pwl_transformer.h"
#include <assert.h>
#include <algorithm>
#include <utility>
#include <stack>
#include <vector>
#include <memory>
#include "tbb/tbb.h"
#include "eigen3/Eigen/Dense"
#include "mkldnn.hpp"

namespace {
void Interpolate(const Eigen::Ref<const RMVectorXf> from_point,
                 const Eigen::Ref<const RMVectorXf> to_point,
                 const double ratio, RMVectorXf *out) {
  out->noalias() = ((1.0 - ratio) * from_point) + (ratio * to_point);
}
}  // namespace

class PWLTransformer::ParallelPlaneTransformer {
 public:
  ParallelPlaneTransformer(const PWLTransformer &layer, UPolytope *inout)
      : layer_(layer), inout_(inout), inserted_points_(new NewPointsMemo()) {}

  size_t MaybeIntersectEdge(IntersectionPointMetadata key) const {
    NewPointsMemo::accessor a;
    if (inserted_points_->insert(a, key)) {
      // Returns true if item is new.
      double crossing_ratio =
        layer_.CrossingRatio(inout_->vertex(key.min_index),
                             inout_->vertex(key.max_index),
                             key.face);
      RMVectorXf vertex;
      RMVectorXf combination;

      Interpolate(inout_->vertex(key.min_index),
                  inout_->vertex(key.max_index),
                  crossing_ratio, &vertex);
      Interpolate(inout_->combination(key.min_index),
                  inout_->combination(key.max_index),
                  crossing_ratio, &combination);
      a->second = inout_->AppendVertex(&vertex, &combination);
    }
    return a->second;
  }

  void operator()(PolytopeMetadata &current_split,
                  tbb::parallel_do_feeder<PolytopeMetadata> &feeder) const {
    int polytope = current_split.polytope_index;
    std::vector<size_t> &possible_faces = current_split.remaining_faces;

    while (!possible_faces.empty()) {
      int n_vertices = inout_->n_vertices(polytope);
      int sign = 0, i = -1, j = -1, split_face = -1;
      for (size_t face_i = 0; face_i < possible_faces.size(); face_i++) {
        bool active = false;
        int face = possible_faces[face_i];
        sign = 0;
        for (i = 0; i < n_vertices; i++) {
          int i_sign = layer_.PointSign(inout_->vertex(polytope, i), face);
          if (i_sign == 0) {
            continue;
          }
          if (sign == 0) {
            sign = i_sign;
            continue;
          }
          if (sign != i_sign) {
            sign = i_sign;
            break;
          }
        }
        if (sign == 0 || i == n_vertices) {
          continue;
        }
        active = layer_.IsFaceActive(inout_->vertex(polytope, i - 1),
                                     inout_->vertex(polytope, i),
                                     face);
        for (j = i + 1; j < n_vertices; j++) {
          int j_sign = layer_.PointSign(inout_->vertex(polytope, j), face);
          if (sign < 0 && j_sign > 0) {
            break;
          } else if (sign > 0 && j_sign < 0) {
            break;
          }
        }
        active = active ||
          layer_.IsFaceActive(inout_->vertex(polytope, j - 1),
                              inout_->vertex(polytope, j % n_vertices),
                              face);
        if (!active) {
          continue;
        }
        split_face = face;
        possible_faces.erase(possible_faces.begin() + face_i);
        break;
      }

      if (split_face == -1) {
        // It's all done!
        return;
      }

      // Now, i is the first vertex with a sign of 'sign,' while j is the first
      // vertex with a sign of '\not sign'.
      IntersectionPointMetadata i_metadata =
        IntersectionPointMetadata(inout_->vertex_index(polytope, i - 1),
                                  inout_->vertex_index(polytope, i),
                                  split_face);
      size_t cross_i_index = MaybeIntersectEdge(i_metadata);
      IntersectionPointMetadata j_metadata =
        IntersectionPointMetadata(inout_->vertex_index(polytope, j - 1),
                                  inout_->vertex_index(polytope,
                                                       j % n_vertices),
                                  split_face);
      size_t cross_j_index = MaybeIntersectEdge(j_metadata);

      // First, we add the intersection points to the existing polytope (which
      // will end up being the 'top' one).
      std::vector<size_t> &top_vertices = inout_->vertex_indices(polytope);
      // NOTE: The order of these insertions is important.
      top_vertices.insert(top_vertices.begin() + i, cross_i_index);
      j++;
      top_vertices.insert(top_vertices.begin() + j, cross_j_index);

      // Now we ``steal'' vertices from the top to put in the bottom.
      std::vector<size_t> bottom_vertices;
      bottom_vertices.reserve(3 + (n_vertices - (j - i)));
      for (size_t v = 0, o = 0; v < top_vertices.size(); o++) {
        if (o <= static_cast<size_t>(i) || o >= static_cast<size_t>(j)) {
          bottom_vertices.push_back(top_vertices[v]);
        }
        if (o < static_cast<size_t>(i) || o > static_cast<size_t>(j)) {
          // These should not be in top_vertices.
          top_vertices.erase(top_vertices.begin() + v);
        } else {
          v++;
        }
      }

      size_t new_polytope = inout_->AppendPolytope(&bottom_vertices);
      feeder.add(PolytopeMetadata(new_polytope, possible_faces));
    }
  }

 private:
  const PWLTransformer &layer_;
  UPolytope *inout_;
  // We use a pointer here because the operator() must be const to work with
  // TBB parallel_do, so we cannot directly modify any member variables.
  std::unique_ptr<NewPointsMemo> inserted_points_;
};

bool PWLTransformer::IsFaceActive(Eigen::Ref<const RMVectorXf> from,
                                  Eigen::Ref<const RMVectorXf> to,
                                  const size_t face) const {
  // This is a safe default.
  return true;
}

void PWLTransformer::EnumerateLineIntersections(
        Eigen::Ref<const RMVectorXf> from_point,
        Eigen::Ref<const RMVectorXf> to_point,
        double from_distance, double to_distance,
        std::vector<double> *new_endpoints) const {
  double delta = to_distance - from_distance;

  std::vector<double> crossing_distances;
  for (size_t i = 0; i < n_piece_faces(to_point.size()); i++) {
    if ((PointSign(to_point, i) * PointSign(from_point, i)) < 0 &&
        IsFaceActive(from_point, to_point, i)) {
      // The points lie in different linear regions, so we need to add an
      // endpoint where they cross this face separating the linear regions.
      // This is the distance between from_distance and to_distance
      double crossing_distance = CrossingRatio(from_point, to_point, i);
      new_endpoints->emplace_back(
          from_distance + (crossing_distance * delta));
    }
  }
}

std::vector<double> PWLTransformer::ProposeLineEndpoints(
    const SegmentedLine &line) const {
  const RMMatrixXf &points = line.points();
  size_t n_segments = line.Size() - 1;

  // NOTE(masotoud): we could use a tbb::concurrent_set here to avoid the merge
  // overhead, or std::vector<std::set<>> to avoid the sort overhead, but my
  // guess is that this those will probably come with too much overhead of
  // their own.
  std::vector<std::vector<double>> segment_endpoints(n_segments);

  tbb::parallel_for(size_t(0), n_segments, [&](size_t i) {
    EnumerateLineIntersections(
            points.row(i), points.row(i + 1),
            line.endpoint_ratio(i), line.endpoint_ratio(i + 1),
            &(segment_endpoints[i]));
    std::sort(segment_endpoints[i].begin(), segment_endpoints[i].end());
  });

  // TODO(masotoud): Perhaps we shouldn't flatten, and just let
  // AddEndpointsThresholded take in the multi-dimensional segment_endpoints?
  std::vector<double> endpoints;
  for (auto &single_segment_endpoints : segment_endpoints) {
    endpoints.insert(endpoints.end(), single_segment_endpoints.begin(),
                     single_segment_endpoints.end());
  }
  segment_endpoints.clear();

  return endpoints;
}

void PWLTransformer::TransformUPolytopePlane(UPolytope *inout) const {
  assert(inout->is_counter_clockwise());

  std::vector<PolytopeMetadata> initial_polytopes;
  std::vector<size_t> all_faces;
  for (size_t i = 0; i < n_piece_faces(inout->space_dimensions()); i++) {
    all_faces.push_back(i);
  }
  for (size_t i = 0; i < inout->n_polytopes(); i++) {
    initial_polytopes.emplace_back(i, all_faces);
  }
  ParallelPlaneTransformer parallel_transformer(*this, inout);
  tbb::parallel_do(initial_polytopes.begin(), initial_polytopes.end(),
                   parallel_transformer);
  Compute(&(inout->vertices()));
}

void PWLTransformer::TransformUPolytope(UPolytope *inout) const {
  if (inout->is_counter_clockwise()) {
    return TransformUPolytopePlane(inout);
  }
  throw "No general-dimension transformer yet.";
}
