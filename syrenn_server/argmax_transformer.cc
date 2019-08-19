#include "syrenn_server/argmax_transformer.h"
#include <vector>
#include <memory>
#include <utility>
#include "eigen3/Eigen/Dense"

std::unique_ptr<LayerTransformer> ArgMaxTransformer::Deserialize(
    const syrenn_server::Layer &layer) {
  if (!layer.has_argmax_data()) {
    return nullptr;
  }
  return std::unique_ptr<ArgMaxTransformer>(new ArgMaxTransformer());
}

size_t ArgMaxTransformer::n_piece_faces(size_t dims) const {
  // One face for each [i] = [j] where i \neq j.
  return (dims * (dims - 1)) / 2;
}

// Takes a face index and produces the corresponding i, j for the face (i.e.,
// the face is described by [i] = [j]).
std::pair<size_t, size_t> SquareIndexFromTriangular(const size_t face,
                                                    const size_t dims) {
  // Formula derived & tested here:
  // https://stackoverflow.com/questions/27086195
  size_t i = dims - 2 - std::floor(
      (std::sqrt((-8 * face) + (4 * dims * (dims - 1) - 7))
      / 2.0) - 0.5);
  size_t j = face + i + 1 - (dims * (dims - 1) / 2) +
              ((dims - i) * ((dims - i) - 1)) / 2;
  return std::make_pair(i, j);
  // Iterative version:
  // for (size_t i = 0; i < dims; i++) {
  //   for (size_t j = i + 1; j < dims; j++, f++) {
  //     if (f == face) {
  //       return std::make_pair(i, j);
  //     }
  //   }
  // }
}

double ArgMaxTransformer::CrossingRatio(Eigen::Ref<const RMVectorXf> from,
                                        Eigen::Ref<const RMVectorXf> to,
                                        const size_t face) const {
  size_t dims = to.size();
  std::pair<size_t, size_t> indices = SquareIndexFromTriangular(face, dims);
  size_t i = indices.first;
  size_t j = indices.second;
  // from[i] + t * delta[i] = from[j] + t * delta[j]
  // t * (delta[i] - delta[j]) = from[j] - from[i]
  // t = from[j] - from[i] / (delta[i] - delta[j])
  float i_delta = to(i) - from(i);
  float j_delta = to(j) - from(j);
  return (from(j) - from(i))
          / (i_delta - j_delta);
}

int ArgMaxTransformer::PointSign(Eigen::Ref<const RMVectorXf> point,
                                 const size_t face) const {
  size_t dims = point.size();
  std::pair<size_t, size_t> indices = SquareIndexFromTriangular(face, dims);
  size_t i = indices.first;
  size_t j = indices.second;
  if (point(i) == point(j)) {
    return 0;
  }
  return (point(i) > point(j)) ? +1 : -1;
}

bool ArgMaxTransformer::IsFaceActive(Eigen::Ref<const RMVectorXf> from,
                                  Eigen::Ref<const RMVectorXf> to,
                                  const size_t face) const {
  size_t dims = from.size();
  std::pair<size_t, size_t> indices = SquareIndexFromTriangular(face, dims);
  size_t i = indices.first;
  size_t j = indices.second;
  size_t from_argmax, to_argmax;
  from.maxCoeff(&from_argmax);
  to.maxCoeff(&to_argmax);
  return (from_argmax == i && to_argmax == j) ||
         (from_argmax == j && to_argmax == i);
}

void ArgMaxTransformer::Compute(RMMatrixXf *inout) const {
  RMMatrixXf temp(inout->rows(), 1);
  for (int i = 0; i < inout->rows(); i++) {
    int index = -1;
    inout->row(i).maxCoeff(&index);
    temp(i, 0) = index;
  }
  inout->swap(temp);
  temp.resize(0, 0);
}
