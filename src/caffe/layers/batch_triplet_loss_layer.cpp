#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchTripletLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(this->layer_param_.loss_weight_size(), 1);
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }

  // accuracy do not contribute to the loss
  // debug information do not contribute to the loss
  for (int i=this->layer_param_.loss_weight_size(); i<top.size(); ++i) {
    this->layer_param_.add_loss_weight(Dtype(0));
  }
    
  margin_ = this->layer_param_.triplet_loss_param().margin();
  mu_ = this->layer_param_.triplet_loss_param().mu();
}

template <typename Dtype>
void BatchTripletLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);      // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);    // average loss over all triplets
  top[1]->Reshape(loss_shape);    // average accuracy rate of all triplets
  if (top.size() == 3) {
    top[2]->Reshape(1, 1, 1, 3);    // 0: average loss over sampled triplets
                                    // 1: number of possible triplets
                                    // 2: number of sampled triplets
  }

  int num = bottom[0]->num();
  dist_.Reshape(num, num, 1, 1);
  norm_.Reshape(num, 1, 1, 1);
}


template <typename Dtype>
void BatchTripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the pairwise distances.
  const Dtype* feat_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* loss_data = top[0]->mutable_cpu_data();
  Dtype* accy_data = top[1]->mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  /**
   * Computing the pairwise Euclidean distance
   */
  Dtype* norm_data = norm_.mutable_cpu_data();
  memset(norm_data, 0, norm_.count() * sizeof(norm_data[0]));
  triplets_.clear();

  Dtype* dist_data = dist_.mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, num, dim, Dtype(-2),
      feat_data, feat_data, Dtype(0), dist_data);

  for (int i=0; i<num; ++i) {
    norm_data[i] = -0.5 * dist_.data_at(i, i, 0, 0);
  }

  for (int i=0; i<num; ++i) {
    dist_data = dist_.mutable_cpu_data() + dist_.offset(i);
    for (int j=0; j<num; ++j) {
      dist_data[j] += (norm_data[i] + norm_data[j]);
    }
  }

  /**
   * Find boundary of different classes.
   * A batch is composed by small groups with items belongs to the same class.
   *  e.g. gruop size is 3, batch size is 15:
   *    1 1 1 2 2 2 3 3 3 1 1 1 4 4 4
   */
  vector<int> boundary;
  Dtype prev = Dtype(-1);
  for (int i=0; i<num; ++i) {
    if (prev != label[i]) {
      boundary.push_back(i);
      prev = label[i];
    }
  }
  boundary.push_back(num);

  /**
   * Sampling triplets and computing the loss
   */
  Dtype smp_loss = 0.0;
  Dtype avg_loss = 0.0;
  int num_tri = 0;
  int num_err = 0;
  Dtype rank_loss;
  Dtype cur_loss;
  Dtype pos_dist;
  Dtype neg_dist;
  Dtype one_minus_mu = Dtype(1) - mu_;
  // classes
  for (int c=0; c<boundary.size()-1; c++) {
    // query
    for (int i=boundary[c]; i<boundary[c+1]; ++i) {
      const Dtype * dist_data  = dist_.cpu_data() + dist_.offset(i);
      // positive
      for (int j=boundary[c]; j<boundary[c+1]; ++j) {
        if (i == j) {
          continue;
        }
        pos_dist = dist_data[j];

        // negative groups
        for (int m=0; m<boundary.size()-1; m++) {
          if (label[boundary[m]] == label[i]) {
            continue;
          }
          // negative
          for (int k=boundary[m]; k<boundary[m+1]; ++k) {
            ++num_tri;
            neg_dist = dist_data[k];
            // DLOG(INFO) << "\t" << pos_dist << "\t" << neg_dist;
            rank_loss = margin_ + pos_dist - neg_dist;
            num_err += (pos_dist >= neg_dist);
            if (rank_loss > 0) {
              cur_loss = rank_loss * mu_ + pos_dist * one_minus_mu;
              avg_loss += cur_loss;
              if (neg_dist > pos_dist) {
                smp_loss += cur_loss;
                triplets_.push_back(Triplet(i, j, k));
              }
            }
          } // end of negative
        } // end of negative groups
      } // end of positive
    } // end of query
  } // end of classes

  int num_smp = triplets_.size();
  // average loss among all triplets
  loss_data[0] = num_tri > 0 ? avg_loss / num_tri : 0;
  // average accuracy among all triplets
  accy_data[0] = Dtype(1) - (num_tri > 0 ? Dtype(num_err) / num_tri : 0);
  if (top.size() == 3) {
    Dtype* debug_data = top[2]->mutable_cpu_data();
    // average loss among selected triplets
    debug_data[0] = num_smp > 0 ? smp_loss / num_smp : 0;
    // number of triplets
    debug_data[1] = num_tri;
    // number of sampled triplets
    debug_data[2] = num_smp;
  }
}

template <typename Dtype>
void BatchTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Blob<Dtype>* feat = bottom[0];
    const Dtype* feat_data = feat->cpu_data();
    Dtype* diff_data = feat->mutable_cpu_diff();
    int count = feat->count();
    int num = feat->num();
    int dim = count / num;
    memset(diff_data, 0, count * sizeof(diff_data[0]));

    Dtype scale1 = Dtype(2) / triplets_.size();
    Dtype scale2 = Dtype(2) / triplets_.size() * mu_;
    Dtype scale0 = scale1 - scale2;
    for (int i=0; i<triplets_.size(); ++i) {
      int qry_offset = feat->offset(triplets_[i].first_);
      int pos_offset = feat->offset(triplets_[i].second_);
      int neg_offset = feat->offset(triplets_[i].third_);

      caffe_cpu_axpby(dim, +scale0, feat_data + qry_offset, Dtype(1), diff_data + qry_offset);
      caffe_cpu_axpby(dim, +scale2, feat_data + neg_offset, Dtype(1), diff_data + qry_offset);
      caffe_cpu_axpby(dim, -scale1, feat_data + pos_offset, Dtype(1), diff_data + qry_offset);

      caffe_cpu_axpby(dim, +scale1, feat_data + pos_offset, Dtype(1), diff_data + pos_offset);
      caffe_cpu_axpby(dim, -scale1, feat_data + qry_offset, Dtype(1), diff_data + pos_offset);

      caffe_cpu_axpby(dim, +scale2, feat_data + qry_offset, Dtype(1), diff_data + neg_offset);
      caffe_cpu_axpby(dim, -scale2, feat_data + neg_offset, Dtype(1), diff_data + neg_offset);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchTripletLossLayer);
#endif

INSTANTIATE_CLASS(BatchTripletLossLayer);
REGISTER_LAYER_CLASS(BatchTripletLoss);

}  // namespace caffe
