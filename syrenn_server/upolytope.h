#ifndef SYRENN_SYRENN_SERVER_UPOLYTOPE_H_
#define SYRENN_SYRENN_SERVER_UPOLYTOPE_H_

#include <memory>
#include <vector>
#include "tbb/tbb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_proto/syrenn.grpc.pb.h"

// Represents a union of V-Polytopes. Each vertex has a "post" and "pre"
// representation; the "post" representation is referred to as the "vertex" and
// the "pre" representation is referred to as the "combination" because it
// represents a convex combination of the pre-transformation vertices.
//
// OPTIMIZATION NOTES: Particularly optimized for the case where many of the
// polytopes share vertices. This implementation stores all vertices
// contiguously in memory, and each polytope is simply a list of indices into
// that contiguous block. This may not be the best choice for all scenarios,
// especially if the block of vertices does not fit in memory.
//
// MULTI-THREADING: Designed for multi-threaded transformer algorithms. In
// particular, new vertices are transparently added to a ``pending'' vector,
// which is merged into the main block when the vertices() method is called.
// Refer to the per-method comments for more information on how to use the
// class in a multi-threaded environment. For the most part, as long as you
// only call "vertices" in a single-threaded environment and assign at most one
// thread per polytope, it should be hard to break things.
class UPolytope {
 public:
  UPolytope(RMMatrixXf *vertices, size_t subspace_dimensions,
            std::vector<std::vector<size_t>> polytopes);

  static UPolytope Deserialize(const syrenn_server::UPolytope &upolytope);
  syrenn_server::UPolytope Serialize() const;

  // Returns a mutable reference to a Matrix containing all vertices.
  // MULTI-THREADING: This method should *NOT* be called when multiple threads
  // may be simultaneously accessing the UPolytope instance; instead, use the
  // vertex and AppendVertex methods. It will serially collapse pending_* into
  // vertices and combinations before returning.
  RMMatrixXf &vertices();
  RMMatrixXf &combinations();

  // Returns a mutable reference to the vector of vertices for a particular
  // polytope.
  // MULTI-THREADING: This method *MAY* be called when multiple threads are
  // simultaneously accessing the UPolytope instance, _as long as_ no two
  // threads are accessing the vertex_indices for the same polytope.
  std::vector<size_t> &vertex_indices(size_t polytope);
  // Returns true iff the UPolytope lives in a two-dimensional subspace and its
  // vertices are in counter-clockwise orientation. (NOTE that in this
  // implementation, it is assumed that all 2D UPolytopes are in CCW
  // orientation).
  // MULTI-THREADING: Always safe.
  bool is_counter_clockwise() const;
  // Returns the number of dimensions of the space that the polytope lives in.
  // This is the same as the number of components of each vertex.
  // MULTI-THREADING: Always safe.
  size_t space_dimensions() const;
  // Returns the number of convex polytopes in this UPolytope.
  // MULTI-THREADING: Always safe.
  size_t n_polytopes() const;
  // Returns the number of vertices in a particular polytope.
  // MULTI-THREADING: Always safe.
  size_t n_vertices(size_t polytope) const;
  // Returns the raw index of the @vertexth vertex in the @polytopeth polytope.
  // MULTI-THREADING: Always safe.
  size_t vertex_index(size_t polytope, size_t vertex) const;
  // Returns an (immutable) reference to the vertex indexed by @raw_index.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> vertex(size_t raw_index) const;
  // Returns an (immutable) reference to the @vertexth vertex in the
  // @polytopeth polytope.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> vertex(size_t polytope, size_t vertex) const;
  // Returns an (immutable) reference to the combination indexed by @raw_index.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> combination(size_t raw_index) const;
  // Returns an (immutable) reference to the @vertexth combination in the
  // @polytopeth polytope.
  // MULTI-THREADING: Always safe.
  Eigen::Ref<const RMVectorXf> combination(size_t polytope,
                                           size_t vertex) const;
  // Appends @vertex and @combination to the list of vertices/combinations,
  // returning their indices.
  // NOTE: This function is *DESTRUCTIVE* to @vertex and @combination.
  // MULTI-THREADING: Safe to call in a multi-threaded evironment. Acquires a
  // write lock on @pending_vertices_ and @pending_combinations_.
  size_t AppendVertex(RMVectorXf *vertex, RMVectorXf *combination);
  // Appends @vertex_indices to the list of polytopes, returning the new
  // polytope's index.
  // NOTE: This function is *DESTRUCTIVE* to @vertex_indices.
  // MULTI-THREADING: Safe to call in a multi-threaded environment. Acquires a
  // write lock on @polytopes_.
  size_t AppendPolytope(std::vector<size_t> *vertex_indices);

 private:
  void FlushPending();

  RMMatrixXf vertices_;
  RMMatrixXf combinations_;

  size_t subspace_dimensions_;
  tbb::concurrent_vector<std::vector<size_t>> polytopes_;

  struct PendingVertex {
    PendingVertex(RMVectorXf *vertex, RMVectorXf *combination)
        : vertex(0), combination(0) {
      this->vertex.swap(*vertex);
      this->combination.swap(*combination);
    }
    RMVectorXf vertex;
    RMVectorXf combination;
  };
  tbb::concurrent_vector<PendingVertex> pending_;
};

#endif  // SYRENN_SYRENN_SERVER_UPOLYTOPE_H_
