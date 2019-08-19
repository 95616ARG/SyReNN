#ifndef SYRENN_SYRENN_SERVER_PWL_TRANSFORMER_H_
#define SYRENN_SYRENN_SERVER_PWL_TRANSFORMER_H_

#include <algorithm>
#include <string>
#include <vector>
#include "tbb/tbb.h"
#include "syrenn_server/segmented_line.h"
#include "syrenn_server/upolytope.h"
#include "syrenn_server/transformer.h"

// Holds metadata about a particular polytopes during the transformation
// process.
struct PolytopeMetadata {
  PolytopeMetadata(size_t polytope_index, std::vector<size_t> remaining_faces)
      : polytope_index(polytope_index), remaining_faces(remaining_faces) {}
  // The index of the polytope into the UPolytope.
  size_t polytope_index;
  // The yet-to-be-split-on faces.
  std::vector<size_t> remaining_faces;
};

// Holds metadata about a vertex added to the UPolytope during the
// transformation process. It is used by NewPointsMemo to ensure that duplicate
// vertices are not added to the UPolytope (when easily avoidable).
struct IntersectionPointMetadata {
  IntersectionPointMetadata(size_t index_from, size_t index_to, size_t face)
      : min_index(std::min(index_from, index_to)),
        max_index(std::max(index_from, index_to)), face(face) {}
  size_t min_index;
  size_t max_index;
  size_t face;
  bool operator==(const IntersectionPointMetadata &other) const {
    return min_index == other.min_index &&
           max_index == other.max_index &&
           face == other.face;
  }
};

// https://software.intel.com/en-us/node/506077
struct IntersectionPointHashCompare {
    static size_t hash(const IntersectionPointMetadata& x) {
      size_t ret = 0;
      hash_combine(&ret, x.min_index, x.max_index, x.face);
      return ret;
    }
    //! True if strings are equal
    static bool equal(const IntersectionPointMetadata& x,
                      const IntersectionPointMetadata& y) {
        return x == y;
    }
};

using NewPointsMemo =
    tbb::concurrent_hash_map<IntersectionPointMetadata, size_t,
                             IntersectionPointHashCompare>;

// Abstract class implementing line/plane transformers for arbitrary
// piecewise-linear functions.
//
// Instantiable child classes must implement:
// - n_piece_faces
// - CrossingRatio
// - PointSign
// - Compute
//
// Note that the implementations of TransformLineInPlace and TransformPlane
// here are only well-optimized for the case of coefficient-wise PWL functions.
// For more complex PWL functions (eg. MaxPool), we suggest custom
// implementations (cf. maxpool_transformer.[h|cc]).
class PWLTransformer : public LayerTransformer {
  // This is a TBB construct that does the actual transformation for 2D
  // polytopes.
  class ParallelPlaneTransformer;
 public:
  std::vector<double> ProposeLineEndpoints(
      const SegmentedLine &line) const override;
  void TransformUPolytope(UPolytope *inout) const override;
  std::string layer_type() const override { return "PWL"; }

 protected:
  // Returns the number of faces defining the partitioning of the function.
  virtual size_t n_piece_faces(size_t dims) const = 0;
  // Returns t such that PointSign(from + t * (to - from), face) = 0, i.e. the
  // ratio corresponding to the intersection of (from->to) with the hyperplane
  // indexed by face.
  virtual double CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                               Eigen::Ref<const RMVectorXf> to,
                               const size_t face) const = 0;
  // Should return true if (not only if) the intersection of @from and @to with
  // @face lies on the border between two linear regions. It is safe to always
  // return true. We use this in ArgMaxTransformer where, for example, the face
  // x[1] = x[2] may be crossed, but it doesn't "actually matter" if x[3] is
  // the maximum component (the face is not actually bordering any of the
  // partitioning polytopes at that point).
  virtual bool IsFaceActive(Eigen::Ref<const RMVectorXf> from,
                            Eigen::Ref<const RMVectorXf> to,
                            const size_t face) const;
  // Returns an integer in { -1, 0, +1 } describing which side of face point
  // lies on. A value of 0 must indicate that point lies on the face, while -1
  // and +1 can indicate either face of the plane (as long as consistency is
  // maintaned).
  virtual int PointSign(Eigen::Ref<const RMVectorXf> point,
                        const size_t face) const = 0;

 private:
  void TransformUPolytopePlane(UPolytope *inout) const;
  void EnumerateLineIntersections(
        Eigen::Ref<const RMVectorXf> from_point,
        Eigen::Ref<const RMVectorXf> to_point,
        double from_distance, double to_distance,
        std::vector<double> *new_endpoints) const;
};

#endif  // SYRENN_SYRENN_SERVER_PWL_TRANSFORMER_H_
