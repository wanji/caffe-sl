#ifndef CAFFE_HDML_LOSS_UPPER_BOUND_LAYER_HPP_
#define CAFFE_HDML_LOSS_UPPER_BOUND_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


/**
 * @brief Computes the upper bound of the HDML loss.
 *
 * @param bottom input Blob vector (length 3)
 *   -# @f$ (N \times B \times 1 \times 1) @f$
 *      query feature.
 *   -# @f$ (N \times B \times 1 \times 1) @f$
 *      positive feature.
 *   -# @f$ (N \times B \times 1 \times 1) @f$
 *      negative feature.
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed upper bound of the HDML loss: @f$ E =
 *      @f$
 */
template <typename Dtype>
class HDMLLossUpperBoundLayer : public LossLayer<Dtype> {
 public:
  explicit HDMLLossUpperBoundLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HDMLLossUpperBound"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:
  /// @copydoc HDMLLossUpperBoundLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the hinge loss error gradient w.r.t. the similarities.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /**
   * Unlike most loss layers, in the HDMLLossUpperBoundLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
 private:
  void compute_h(const Blob<Dtype> * qry,
      const Blob<Dtype> * pos, const Blob<Dtype> * neg);
  void compute_infer_loss(const Blob<Dtype> * qry,
      const Blob<Dtype> * pos, const Blob<Dtype> * neg);
  Dtype compute_loss_ub();

  Blob<Dtype> infer_cont_;      // N x D x 3 x 1, cont(i, e_i) in Eq (10)
  Blob<int> infer_qry_buffer_;  // N x D x 3 x 1, a in Eq (10)
  Blob<int> infer_pos_buffer_;  // N x D x 3 x 1, b in Eq (10)
  Blob<int> infer_neg_buffer_;  // N x D x 3 x 1, c in Eq (10)

  Blob<int> h_qry_;             // N x D x 1 x 1, h   in Eq(6)
  Blob<int> h_pos_;             // N x D x 1 x 1, h^+ in Eq(6)
  Blob<int> h_neg_;             // N x D x 1 x 1, h^- in Eq(6)

  Blob<int> g_qry_;             // N x D x 1 x 1, g   in Eq(6)
  Blob<int> g_pos_;             // N x D x 1 x 1, g^+ in Eq(6)
  Blob<int> g_neg_;             // N x D x 1 x 1, g^- in Eq(6)
};



}  // namespace caffe

#endif  // CAFFE_HDML_LOSS_UPPER_BOUND_LAYER_HPP_
