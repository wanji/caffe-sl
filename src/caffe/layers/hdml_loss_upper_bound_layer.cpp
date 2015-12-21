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
void HDMLLossUpperBoundLayer<Dtype>::compute_infer_abc_loss(
    const Blob<Dtype> * qry,
    const Blob<Dtype> * pos,
    const Blob<Dtype> * neg) {
  int num = qry->num();
  int nb = qry->channels();

  for (int k=0; k<num; k++) {
    const Dtype * qry_k = qry->cpu_data() + qry->offset(k * nb);
    const Dtype * pos_k = pos->cpu_data() + pos->offset(k * nb);
    const Dtype * neg_k = neg->cpu_data() + neg->offset(k * nb);

    for (int i=0; i<nb; ++i) {
      // index(ei) starts from -1 to +1
      Dtype * p_loss = this->infer_abc_loss_.mutable_cpu_data() + \
                       this->infer_abc_loss_.offset(k * nb * 3 + i * 3 + 1);
      int * p_qry_buf = this->infer_qry_buffer_.mutable_cpu_data() + \
                        this->infer_qry_buffer_.offset(k * nb * 3 + i * 3 + 1);
      int * p_pos_buf = this->infer_pos_buffer_.mutable_cpu_data() + \
                        this->infer_pos_buffer_.offset(k * nb * 3 + i * 3 + 1);
      int * p_neg_buf = this->infer_neg_buffer_.mutable_cpu_data() + \
                        this->infer_neg_buffer_.offset(k * nb * 3 + i * 3 + 1);

      for (int ei=-1; ei<=1; ++ei) {
        p_loss[ei] = -this->c_max_flt_;
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
Dtype HDMLLossUpperBoundLayer<Dtype>::compute_loss_ub() {
  int num = this->infer_abc_loss_.num();
  int nb = this->infer_abc_loss_.channels();
  int step = nb * 2 + 1;
  Dtype * m_table = new Dtype[step * nb];
  int * choice_table = new int[step * nb];
  Dtype * p_m_old;
  Dtype * p_m_cur;
  int * p_c_old;
  int * p_c_cur;
  Dtype cur_loss;
  Dtype loss_ub = 0.0;

  for (int k=0; k<num; ++k) {
    /**
     * Compute the table
     */
    const Dtype * p_loss = this->infer_abc_loss_.cpu_data() + \
                     this->infer_abc_loss_.offset(k * nb * 3 + 1);
    p_m_cur = m_table + nb;
    p_c_cur = choice_table + nb;

    // the first row of m_table and choice_table
    for (int i=-1; i<=1; i++) {
      p_c_cur[i] = i;
      p_m_cur[i] = p_loss[i];
    }
    p_m_old = p_m_cur;
    p_c_old = p_c_cur;

    // the sencond and rest rows of m_table and choice_table
    for (int i=1; i<nb; ++i) {
      p_m_cur = p_m_old + step;
      p_c_cur = p_c_old + step;
      p_loss += 3;
      for (int m=-i-1; m<=+i+1; ++m) {
        p_m_cur[m] = -this->c_max_flt_;
        for (int l=std::max(m-1, -i); l<=std::min(m+1, i); ++l) {
          cur_loss = p_m_old[l] + p_loss[m-l];
          if (cur_loss > p_m_cur[m]) {
            p_m_cur[m] = cur_loss;
            p_c_cur[m] = m - l;
          }
        }
      }
      p_m_old = p_m_cur;
      p_c_old = p_c_cur;
    }

    /**
     * find the path
     */
    int max_m = -nb;
    for (int m=-nb+1; m<=nb; ++m) {
      if (p_m_cur[m] > p_m_cur[max_m]) {
        max_m = m;
      }
    }
    loss_ub += p_m_cur[max_m];
    int * p_g_qry_ = this->g_qry_.mutable_cpu_data() + this->g_qry_.offset(k * nb);
    int * p_g_pos_ = this->g_pos_.mutable_cpu_data() + this->g_pos_.offset(k * nb);
    int * p_g_neg_ = this->g_neg_.mutable_cpu_data() + this->g_neg_.offset(k * nb);
    int * p_qry_buf = this->infer_qry_buffer_.mutable_cpu_data() + this->infer_qry_buffer_.offset(k * nb * 3 + 1);
    int * p_pos_buf = this->infer_pos_buffer_.mutable_cpu_data() + this->infer_pos_buffer_.offset(k * nb * 3 + 1);
    int * p_neg_buf = this->infer_neg_buffer_.mutable_cpu_data() + this->infer_neg_buffer_.offset(k * nb * 3 + 1);
    int ei;
    for (int i=nb-1; i>=0; --i) {
      ei = p_c_cur[max_m];
      max_m -= ei;
      p_c_cur -= step;
      p_g_qry_[i] = p_qry_buf[i * 3 + ei];
      p_g_pos_[i] = p_pos_buf[i * 3 + ei];
      p_g_neg_[i] = p_neg_buf[i * 3 + ei];
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
  int nb = bottom[0]->channels();
  
  this->infer_qry_buffer_.Reshape(num, nb, 3, 1);
  this->infer_pos_buffer_.Reshape(num, nb, 3, 1);
  this->infer_neg_buffer_.Reshape(num, nb, 3, 1);
  this->infer_abc_loss_.Reshape(num, nb, 3, 1);
  this->h_qry_.Reshape(num, nb, 1, 1);
  this->g_qry_.Reshape(num, nb, 1, 1);
  this->h_pos_.Reshape(num, nb, 1, 1);
  this->g_pos_.Reshape(num, nb, 1, 1);
  this->h_neg_.Reshape(num, nb, 1, 1);
  this->g_neg_.Reshape(num, nb, 1, 1);
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* qry = bottom[0];   // ->cpu_data();
  const Blob<Dtype>* pos = bottom[1];   // ->cpu_data();
  const Blob<Dtype>* neg = bottom[2];   // ->cpu_data();

  int num = qry->num();
  int nb = qry->channels();
  for (int k=0; k<num; k++) {
    const Dtype * p_qry = qry->cpu_data() + qry->offset(k * nb);
    const Dtype * p_pos = pos->cpu_data() + pos->offset(k * nb);
    const Dtype * p_neg = neg->cpu_data() + neg->offset(k * nb);

    int * p_h_qry = this->h_qry_.mutable_cpu_data() + this->h_qry_.offset(k * nb);
    int * p_h_pos = this->h_pos_.mutable_cpu_data() + this->h_pos_.offset(k * nb);
    int * p_h_neg = this->h_neg_.mutable_cpu_data() + this->h_neg_.offset(k * nb);
    for (int i=0; i<nb; i++) {
      p_h_qry[i] = p_qry[i] > 0 ? 1 : -1;
      p_h_pos[i] = p_pos[i] > 0 ? 1 : -1;
      p_h_neg[i] = p_neg[i] > 0 ? 1 : -1;
    }
  }

  this->compute_infer_abc_loss(qry, pos, neg);

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = this->compute_loss_ub();
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int num = bottom[0]->num();
    int nb = bottom[0]->channels();
    for (int k=0; k<num; ++k) {
      Dtype* qry_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(k * nb);
      Dtype* pos_diff = bottom[1]->mutable_cpu_diff() + bottom[1]->offset(k * nb);
      Dtype* neg_diff = bottom[2]->mutable_cpu_diff() + bottom[2]->offset(k * nb);

      const int * p_h_qry = this->h_qry_.cpu_data() + this->h_qry_.offset(k * nb);
      const int * p_h_pos = this->h_pos_.cpu_data() + this->h_pos_.offset(k * nb);
      const int * p_h_neg = this->h_neg_.cpu_data() + this->h_neg_.offset(k * nb);

      const int * p_g_qry = this->g_qry_.cpu_data() + this->g_qry_.offset(k * nb);
      const int * p_g_pos = this->g_pos_.cpu_data() + this->g_pos_.offset(k * nb);
      const int * p_g_neg = this->g_neg_.cpu_data() + this->g_neg_.offset(k * nb);
      for (int i=0; i<nb; ++i) {
        qry_diff[i] = p_h_qry[i] - p_g_qry[i];
        pos_diff[i] = p_h_pos[i] - p_g_pos[i];
        neg_diff[i] = p_h_neg[i] - p_g_neg[i];
      }
    }
  }
}

INSTANTIATE_CLASS(HDMLLossUpperBoundLayer);
REGISTER_LAYER_CLASS(HDMLLossUpperBound);

}  // namespace caffe
