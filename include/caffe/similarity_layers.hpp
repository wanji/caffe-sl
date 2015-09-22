#ifndef CAFFE_LOSS_LAYERS_HPP_
#define CAFFE_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
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

/**
 * @brief Computes the Euclidean Similarity @f$
 *          S = - \left| \left| q - p \right| \right|_2^2 @f$.
 */
template <typename Dtype>
class EuclideanSimilarityLayer : public SimilarityLayer<Dtype> {
 public:
  explicit EuclideanSimilarityLayer(const LayerParameter& param)
     : SimilarityLayer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "EuclideanSimilarity"; }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> diff_;
};

/**
 * @brief Computes the Euclidean Similarity @f$
 *          S = - \left| \left| q - p \right| \right|_2^2 @f$.
 */
template <typename Dtype>
class DotProductSimilarityLayer : public SimilarityLayer<Dtype> {
 public:
  explicit DotProductSimilarityLayer(const LayerParameter& param)
     : SimilarityLayer<Dtype>(param) {}
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYERS_HPP_
