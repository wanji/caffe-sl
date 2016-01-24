#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief An interface for Layer%s that take two Blob%s as input and output a
 *        singleton Blob representing the similarity.
 */
template <typename Dtype>
class SimilarityLayer : public Layer<Dtype> {
 public:
  explicit SimilarityLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually can backpropagate to both inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYERS_HPP_
