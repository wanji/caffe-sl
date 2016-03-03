#ifndef CAFFE_EUCLIDEAN_SIMILARITY_LAYER_HPP_
#define CAFFE_EUCLIDEAN_SIMILARITY_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/similarity_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the Euclidean Similarity @f$
 *          S = - \left| \left| q - p \right| \right|_2^2 @f$.
 */
template <typename Dtype>
class EuclideanSimilarityLayer : public SimilarityLayer<Dtype> {
 public:
  explicit EuclideanSimilarityLayer(const LayerParameter& param)
     : SimilarityLayer<Dtype>(param) {}
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


}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_SIMILARITY_LAYER_HPP_


