#ifndef CAFFE_NAIVE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_NAIVE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {


/**
 * @brief Computes the hinge loss for pair-wise learning to rank task.
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (3N \times D \times 1 \times 1) @f$
 *      feature @f$ qry/pos/neg (N \times D \times 1 \times 1) @f$.
 * @param top output Blob vector (length 2)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed hinge loss: @f$ E =
 *        \frac{1}{N} \sum\limits_{n=1}^N
 *        [\max(0, margin - S(q, p^+) + S(q, p^-))]
 *      @f$
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed ranking accuracy
 */
template <typename Dtype>
class NaiveTripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit NaiveTripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NaiveTripletLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  /// @copydoc NaiveTripletLossLayer 
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the hinge loss error gradient w.r.t. the similarities.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /**
   * Unlike most loss layers, in the PairwiseRankingLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /// The internal SimilarityLayer.
  Blob<Dtype> qry_feat_;
  Blob<Dtype> pos_feat_;
  Blob<Dtype> neg_feat_;
  Blob<Dtype> pos_sim_;
  Blob<Dtype> neg_sim_;

  shared_ptr<Layer<Dtype> > split_layer_;
  vector<Blob<Dtype>*> split_bottom_vec_;
  vector<Blob<Dtype>*> split_top_vec_;

  shared_ptr<Layer<Dtype> > pos_sim_layer_;
  vector<Blob<Dtype>*> pos_sim_bottom_vec_;
  vector<Blob<Dtype>*> pos_sim_top_vec_;

  shared_ptr<Layer<Dtype> > neg_sim_layer_;
  vector<Blob<Dtype>*> neg_sim_bottom_vec_;
  vector<Blob<Dtype>*> neg_sim_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_NAIVE_TRIPLET_LOSS_LAYER_HPP_
