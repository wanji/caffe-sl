#ifndef CAFFE_PAIR_WISE_RANKING_LOSS_LAYER_HPP_
#define CAFFE_PAIR_WISE_RANKING_LOSS_LAYER_HPP_

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
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      similarity @f$ S(q, p^+) @f$.
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      similarity @f$ S(q, p^-) @f$.
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed hinge loss: @f$ E =
 *        \frac{1}{N} \sum\limits_{n=1}^N
 *        [\max(0, margin - S(q, p^+) + S(q, p^-))]
 *      @f$
 */
template <typename Dtype>
class PairwiseRankingLossLayer : public LossLayer<Dtype> {
 public:
  explicit PairwiseRankingLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "PairwiseRankingLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc PairwiseRankingLossLayer
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
};



}  // namespace caffe

#endif  // CAFFE_PAIR_WISE_RANKING_LOSS_LAYER_HPP_

