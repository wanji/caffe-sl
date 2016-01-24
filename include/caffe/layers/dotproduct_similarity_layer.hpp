#ifndef CAFFE_DOTPRODUCT_SIMILARITY_LAYER_HPP_
#define CAFFE_DOTPRODUCT_SIMILARITY_LAYER_HPP_

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
 * @brief Computes the Dot-Product Similarity @f$ S = q^\top  p  @f$.
 */
template <typename Dtype>
class DotProductSimilarityLayer : public SimilarityLayer<Dtype> {
 public:
  explicit DotProductSimilarityLayer(const LayerParameter& param)
     : SimilarityLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "DotProductSimilarity"; }
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

#endif  // CAFFE_DOTPRODUCT_SIMILARITY_LAYER_HPP_


