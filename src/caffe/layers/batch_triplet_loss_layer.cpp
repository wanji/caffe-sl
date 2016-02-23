#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/batch_triplet_loss_layer.hpp"
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
  if (top.size() == 3) {
    LOG(INFO) << "Loss debug information:";
    LOG(INFO) << "\t0: average rank loss over sampled triplets";
    LOG(INFO) << "\t1: average rank loss over all triplets    ";
    LOG(INFO) << "\t2: average pair loss                      ";
    LOG(INFO) << "\t3: number of possible triplets            ";
    LOG(INFO) << "\t4: number of sampled triplets             ";
  }
    
  margin_ = this->layer_param_.triplet_loss_param().margin();
  mu_ = this->layer_param_.triplet_loss_param().mu();
}

template <typename Dtype>
void BatchTripletLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);      // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);    // average loss (rank_loss + pair_loss)
  top[1]->Reshape(loss_shape);    // average accuracy rate of all triplets
  if (top.size() == 3) {
    top[2]->Reshape(1, 1, 1, 5);
  }

  int num = bottom[0]->num();
  dist_.Reshape(num, num, 1, 1);
  norm_.Reshape(num, 1, 1, 1);
  aggregator_.reset(new SyncedMemory(num * num * sizeof(Dtype)));
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
  bool sample = this->layer_param_.triplet_loss_param().sample();

  /**
   * Computing the pairwise Euclidean distance
   */
  Dtype* norm_data = norm_.mutable_cpu_data();
  memset(norm_data, 0, norm_.count() * sizeof(norm_data[0]));
  triplets_.clear();
  pos_pairs_.clear();

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
  Dtype pair_loss = Dtype(0);
  Dtype rank_loss = Dtype(0);
  Dtype smp_rank_loss = Dtype(0);
  int num_tri = 0;
  int num_err = 0;
  Dtype cur_rank_loss;
  Dtype pos_dist;
  Dtype neg_dist;
  Dtype one_minus_mu = Dtype(1) - mu_;

  if (one_minus_mu > Dtype(0)) {
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
          pair_loss += dist_data[j];
          pos_pairs_.push_back(make_pair<int, int>(i, j));
        }
      }
    }
  }
  int num_pair = pos_pairs_.size();
  pair_loss = num_pair > 0 ? pair_loss / num_pair : 0;

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
            cur_rank_loss = margin_ + pos_dist - neg_dist;
            // LOG(INFO) << cur_rank_loss;
            num_err += (pos_dist >= neg_dist);
            if (cur_rank_loss > 0) {
              rank_loss += cur_rank_loss;
              if (!sample || neg_dist > pos_dist) {
                smp_rank_loss += cur_rank_loss;
                triplets_.push_back(Triplet(i, j, k));
              }
            }
          } // end of negative
        } // end of negative groups
      } // end of positive
    } // end of query
  } // end of classes
  rank_loss = num_tri > 0 ? rank_loss / num_tri : 0;

  // average loss among all triplets
  loss_data[0] = rank_loss * mu_ + pair_loss * one_minus_mu;
  // average accuracy among all triplets
  accy_data[0] = Dtype(1) - (num_tri > 0 ? Dtype(num_err) / num_tri : 0);
  if (top.size() == 3) {
    int num_smp = triplets_.size();
    Dtype* debug_data = top[2]->mutable_cpu_data();

    // 0: average rank loss over sampled triplets
    debug_data[0] = num_smp > 0 ? smp_rank_loss / num_smp : 0;
    // 1: average rank loss over all triplets
    debug_data[1] = rank_loss;
    // 2: average pair loss
    debug_data[2] = pair_loss;
    // 3: number of possible triplets
    debug_data[3] = num_tri;
    // 4: number of sampled triplets
    debug_data[4] = num_smp;
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
    Dtype* feat_diff = feat->mutable_cpu_diff();
    int count = feat->count();
    int num = feat->num();
    int dim = count / num;
    int agg_step = num * sizeof(Dtype);
    Dtype * agg_data = (Dtype *)aggregator_->mutable_cpu_data();
    caffe_memset(num * agg_step, 0, agg_data);

    Dtype scale1 = Dtype(2) / triplets_.size() * mu_;
    for (int i=0; i<triplets_.size(); ++i) {
      int qry_id = triplets_[i].first_;
      int pos_id = triplets_[i].second_;
      int neg_id = triplets_[i].third_;

      agg_data[qry_id * num + neg_id] += scale1;
      agg_data[qry_id * num + pos_id] -= scale1;

      agg_data[pos_id * num + pos_id] += scale1;
      agg_data[pos_id * num + qry_id] -= scale1;

      agg_data[neg_id * num + qry_id] += scale1;
      agg_data[neg_id * num + neg_id] -= scale1;
    }

    Dtype scale2 = Dtype(2) / pos_pairs_.size() * (Dtype(1) - mu_);
    for (int i=0; i<pos_pairs_.size(); ++i) {
      int qry_id = pos_pairs_[i].first;
      int pos_id = pos_pairs_[i].second;

      agg_data[qry_id * num + qry_id] += scale2;
      agg_data[qry_id * num + pos_id] -= scale2;

      agg_data[pos_id * num + pos_id] += scale2;
      agg_data[pos_id * num + qry_id] -= scale2;
    }

    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, num,
        Dtype(1), agg_data, feat_data, Dtype(0), feat_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchTripletLossLayer);
#endif

INSTANTIATE_CLASS(BatchTripletLossLayer);
REGISTER_LAYER_CLASS(BatchTripletLoss);

}  // namespace caffe
