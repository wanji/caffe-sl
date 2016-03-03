#ifndef CAFFE_LP_NORM_LAYER_HPP_
#define CAFFE_LP_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Normalizes the blob to have unit Lp norm.
 */
template <typename Dtype>
class L2NormLayer : public NeuronLayer<Dtype> {
 public:
  explicit L2NormLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param), inv_lpnorm_(NULL) {}
  virtual ~L2NormLayer() {
    if (inv_lpnorm_ != NULL) {
      delete [] inv_lpnorm_;
    }
  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L2Norm"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc L2NormLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype * inv_lpnorm_;
};

}  // namespace caffe

#endif  // CAFFE_LP_NORM_LAYER_HPP_

