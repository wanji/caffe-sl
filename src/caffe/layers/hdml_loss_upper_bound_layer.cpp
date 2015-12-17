#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void compute_max_abc_loss(const Blob<Dtype> * qry,
    const Blob<Dtype> * pos, const Blob<Dtype> * neg) {
  int num = qry->num();
  int nb = qry->channels();

  for (int k=0; k<num; k++) {
    const Dtype * qry_k = qry->cpu_data() + qry->offset(k * nb);
    const Dtype * pos_k = pos->cpu_data() + pos->offset(k * nb);
    const Dtype * neg_k = neg->cpu_data() + neg->offset(k * nb);

    for (int i=0; i<nb; ++i) {
      Dtype * p_loss = max_abc_loss_.mutable_cpu_data() + max_abc_loss_.offset(k * nb * 3 + i * 3 + 1);
      Dtype * p_qry_buf = max_qry_buffer_.mutable_cpu_data() + max_qry_buffer_.offset(k * nb * 3 + i * 3 + 1);
      Dtype * p_pos_buf = max_pos_buffer_.mutable_cpu_data() + max_pos_buffer_.offset(k * nb * 3 + i * 3 + 1);
      Dtype * p_neg_buf = max_neg_buffer_.mutable_cpu_data() + max_neg_buffer_.offset(k * nb * 3 + i * 3 + 1);
      for (int ei=-1; ei=<=1; ++ei) {
        // Dtype max_cont = \
        //                  (qry_k[i] > 0 ? -qry_k[i] : qry_k[i]) + \
        //                  (pos_k[i] > 0 ? -pos_k[i] : pos_k[i]) + \
        //                  (neg_k[i] > 0 ? -neg_k[i] : neg_k[i]);
        Dtype p_loss[ei] = -c_max_flt_;
        Dtype cont;
        for (int a=-1; a<=1; a+=2) {
          for (int b=-1; b<=1; b+=2) {
            for (int c=-1; c<=1; c+=2) {
              if ((a != b) - (a != c) != ei) {
                continue;
              }
              cont = a * qry_k[i] + b * pos_k[i] + c * neg_k[i];
              if (cont > p_loss[ei]) {
                p_loss[ei] = cont;
                p_qry_buf[ei] = a;
                p_pos_buf[ei] = b;
                p_neg_buf[ei] = c;
              }
            }
          }
        }
      }
    }
  }
}


template <typename Dtype>
Dtype compute_loss_ub() {
  int num = this->max_abc_loss_->num();
  int nb = this->max_abc_loss_->channels();
  int step = nb * 2 + 1;
  Dtype * m_table = new Dtype[step * nb];
  int * choice_table = new int[step * nb];
  Dtype * p_m;
  Dtype * p_c;
  Dtype cur_loss;

  for (int k=0; k<num; ++k) {
    /**
     * Compute the table
     */
    Dtype * p_loss = this->max_abc_loss_.cpu_data() + this->max_abc_loss_.offset(k * nb * 3 + 1);
    p_m = m_table + nb;
    p_c = choice_table + nb;

    for (int i=-1; i<=1; i++) {
      p_c[i] = i;
      p_m[i] = p_loss[i];
    }

    for (int i=1; i<nb; ++i) {
      p_loss += 3;
      for (int m=-i-1; m<=+i+1; ++m) {
        p_m[step+m] = -c_max_flt_;
        for (int l=std::max(j-1, -i); l<=std::min(j+1, i); ++l) {
          cur_loss = p_m[l] + p_loss[m-l];
          if (cur_loss > p_m[step+m]) {
            p_m[step+m] = cur_loss;
            p_c[step+m] = m - l;
          }
        }
      }
      p_m += step;
      p_c += step;
    }

    /**
     * find the path
     */
    int max_m = -nb;
    for (int m=-nb+1; m<=nb; ++m) {
      if (p_m[m] > p_m[max_m]) {
        max_m = m;
      }
    }
    
  }

  delete [] m_table;
  delete [] choice_table;
  return loss_ub;
}


template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);

  int num = bottom[0]->num();
  int dim = bottom[0]->channels();
  
  this->max_qry_buffer_.Reshape(num, dim, 3, 1);
  this->max_pos_buffer_.Reshape(num, dim, 3, 1);
  this->max_neg_buffer_.Reshape(num, dim, 3, 1);
  this->max_abc_loss_.Reshape(num, dim, 3, 1);
  this->h_qry_.Reshape(num, dim, 1, 1);
  this->g_qry_.Reshape(num, dim, 1, 1);
  this->h_pos_.Reshape(num, dim, 1, 1);
  this->g_pos_.Reshape(num, dim, 1, 1);
  this->h_neg_.Reshape(num, dim, 1, 1);
  this->g_neg_.Reshape(num, dim, 1, 1);
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* qry = bottom[0]->cpu_data();
  const Dtype* pos = bottom[1]->cpu_data();
  const Dtype* neg = bottom[2]->cpu_data();

  Dtype* per_triplet_loss_ub = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;
  for (int i=0; i<count; ++i) {
    per_triplet_loss[i] = std::max(Dtype(0),
        this->layer_param_.pairwise_ranking_loss_param().margin()
        - pos_sim[i] + neg_sim[i]);
    loss[0] += per_triplet_loss[i];
  }
  loss[0] /= count;
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* pos_diff = bottom[0]->mutable_cpu_diff();
    Dtype* neg_diff = bottom[1]->mutable_cpu_diff();
    int count = bottom[0]->count();
    for (int i=0; i<count; ++i) {
      pos_diff[i] = pos_diff[i] ? -1 : 0;
      neg_diff[i] = pos_diff[i] ?  1 : 0;
    }
  }
}

INSTANTIATE_CLASS(HDMLLossUpperBoundLayer);
REGISTER_LAYER_CLASS(HDMLLossUpperBound);

}  // namespace caffe
