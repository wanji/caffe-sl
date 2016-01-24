#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hdml_loss_upper_bound_layer.hpp"

namespace caffe {

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::compute_h(
    const Blob<Dtype> * qry,
    const Blob<Dtype> * pos,
    const Blob<Dtype> * neg) {
  int num = qry->num();
  int nb = qry->channels();
  for (int k=0; k<num; k++) {
    const Dtype * qry_k = qry->cpu_data() + qry->offset(k);
    const Dtype * pos_k = pos->cpu_data() + pos->offset(k);
    const Dtype * neg_k = neg->cpu_data() + neg->offset(k);

    int * p_h_qry = this->h_qry_.mutable_cpu_data() + this->h_qry_.offset(k);
    int * p_h_pos = this->h_pos_.mutable_cpu_data() + this->h_pos_.offset(k);
    int * p_h_neg = this->h_neg_.mutable_cpu_data() + this->h_neg_.offset(k);

    for (int i=0; i<nb; i++) {
      p_h_qry[i] = qry_k[i] > 0 ? 1 : -1;
      p_h_pos[i] = pos_k[i] > 0 ? 1 : -1;
      p_h_neg[i] = neg_k[i] > 0 ? 1 : -1;
    }
  }
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::compute_infer_loss(
    const Blob<Dtype> * qry,
    const Blob<Dtype> * pos,
    const Blob<Dtype> * neg) {
  int num = qry->num();
  int nb = qry->channels();

  for (int k=0; k<num; k++) {
    const Dtype * qry_k = qry->cpu_data() + qry->offset(k);
    const Dtype * pos_k = pos->cpu_data() + pos->offset(k);
    const Dtype * neg_k = neg->cpu_data() + neg->offset(k);

    for (int i=0; i<nb; ++i) {
      // index(ei) starts from -1 to +1
      Dtype * p_cont = this->infer_cont_.mutable_cpu_data() + \
                       this->infer_cont_.offset(k, i, 1);
      int * p_qry_buf = this->infer_qry_buffer_.mutable_cpu_data() + \
                        this->infer_qry_buffer_.offset(k, i, 1);
      int * p_pos_buf = this->infer_pos_buffer_.mutable_cpu_data() + \
                        this->infer_pos_buffer_.offset(k, i, 1);
      int * p_neg_buf = this->infer_neg_buffer_.mutable_cpu_data() + \
                        this->infer_neg_buffer_.offset(k, i, 1);

      for (int ei=-1; ei<=1; ++ei) {
        p_cont[ei] = -1 - qry_k[i] - pos_k[i] - neg_k[i];
      }
      Dtype cont;
      for (int a=-1; a<=1; a+=2) {
        for (int b=-1; b<=1; b+=2) {
          for (int c=-1; c<=1; c+=2) {
            int ei = (a != b) - (a != c);
            cont = a * qry_k[i] + b * pos_k[i] + c * neg_k[i];
            if (cont > p_cont[ei]) {
              p_cont[ei] = cont;
              p_qry_buf[ei] = a;
              p_pos_buf[ei] = b;
              p_neg_buf[ei] = c;
            }
          }
        }
      }
    }
  }
  // this->infer_cont_.print();
  // exit(1);
}


template <typename Dtype>
Dtype HDMLLossUpperBoundLayer<Dtype>::compute_loss_ub() {
  int num = this->infer_cont_.num();
  int nb = this->infer_cont_.channels();
  int step = nb * 2 + 1;

  Dtype * m_table = new Dtype[step * nb];
  Dtype * p_m_old;
  Dtype * p_m_cur;

  int * choice_table = new int[step * nb];
  int * p_c;

  Dtype loss_ub = 0.0;
  Dtype cur_loss;

  for (int k=0; k<num; ++k) {
    /**
     * Compute the table
     */
    const Dtype * p_cont = this->infer_cont_.cpu_data() + \
                           this->infer_cont_.offset(k, 0, 1);
    p_m_cur = m_table + nb;
    p_c = choice_table + nb;

    // the first row of m_table and choice_table
    for (int m=-1; m<=1; m++) {
      p_m_cur[m] = p_cont[m];
      p_c[m] = m;
    }

    // the sencond and rest rows of m_table and choice_table
    for (int i=1; i<nb; ++i) {
      p_m_old = p_m_cur;
      p_m_cur += step;
      p_c += step;
      p_cont += 3;
      for (int m=-i-1; m<=+i+1; ++m) {
        p_m_cur[m] = -1e30;
#if 0
        for (int ei=-1; ei<=1; ++ei) {
          if (m-ei >= -i && m-ei <= i) {
            cur_loss = p_m_old[m-ei] + p_cont[ei];
            if (cur_loss > p_m_cur[m]) {
              p_m_cur[m] = cur_loss;
              p_c[m] = ei;
            }
          }
        }
#else
        for (int l=std::max(m-1, -i); l<=std::min(m+1, i); ++l) {
          cur_loss = p_m_old[l] + p_cont[m-l];
          if (cur_loss > p_m_cur[m]) {
            p_m_cur[m] = cur_loss;
            p_c[m] = m - l;
          }
        }
#endif
      }
    }

    /**
     * find the path
     */
    int max_m = -nb;
    Dtype max_infer = std::max(max_m - 1, 0) + p_m_cur[max_m];  // Eq(7)
    Dtype cur_infer;
    for (int m=-nb+1; m<=nb; ++m) {
      cur_infer = std::max(m - 1, 0) + p_m_cur[m];
      if (cur_infer > max_infer) {
        max_m = m;
        max_infer = cur_infer;
      }
    }
    LOG(INFO) << max_m << '\t' << max_infer;
    loss_ub += max_infer;
    int * p_g_qry_ = this->g_qry_.mutable_cpu_data() + this->g_qry_.offset(k);
    int * p_g_pos_ = this->g_pos_.mutable_cpu_data() + this->g_pos_.offset(k);
    int * p_g_neg_ = this->g_neg_.mutable_cpu_data() + this->g_neg_.offset(k);
    int ei;
    for (int i=nb-1; i>=0; --i) {
      ei = p_c[max_m];
      p_g_qry_[i] = this->infer_qry_buffer_.data_at(k, i, ei+1, 0);
      p_g_pos_[i] = this->infer_pos_buffer_.data_at(k, i, ei+1, 0);
      p_g_neg_[i] = this->infer_neg_buffer_.data_at(k, i, ei+1, 0);
      max_m -= ei;
      p_c -= step;
    }
  }

  delete [] m_table;
  delete [] choice_table;
  return loss_ub;
}


template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }

  int num = bottom[0]->num();
  int nb = bottom[0]->channels();
  
  CHECK_EQ(num, bottom[1]->num());
  CHECK_EQ(num, bottom[2]->num());
  CHECK_EQ(nb, bottom[1]->channels());
  CHECK_EQ(nb, bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->width(), 1);

  this->infer_cont_.Reshape(num, nb, 3, 1);
  this->infer_qry_buffer_.Reshape(num, nb, 3, 1);
  this->infer_pos_buffer_.Reshape(num, nb, 3, 1);
  this->infer_neg_buffer_.Reshape(num, nb, 3, 1);

  this->h_qry_.Reshape(num, nb, 1, 1);
  this->h_pos_.Reshape(num, nb, 1, 1);
  this->h_neg_.Reshape(num, nb, 1, 1);

  this->g_qry_.Reshape(num, nb, 1, 1);
  this->g_pos_.Reshape(num, nb, 1, 1);
  this->g_neg_.Reshape(num, nb, 1, 1);
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Blob<Dtype>* qry = bottom[0];   // ->cpu_data();
  const Blob<Dtype>* pos = bottom[1];   // ->cpu_data();
  const Blob<Dtype>* neg = bottom[2];   // ->cpu_data();

  this->compute_h(qry, pos, neg);
  this->compute_infer_loss(qry, pos, neg);

  Dtype loss_ub = this->compute_loss_ub();
  for (int i=0; i<qry->count(); i++) {
    loss_ub -= std::abs(qry->cpu_data()[i]);
    loss_ub -= std::abs(pos->cpu_data()[i]);
    loss_ub -= std::abs(neg->cpu_data()[i]);
  }
  loss_ub /= qry->num();

  top[0]->mutable_cpu_data()[0] = loss_ub;
}

template <typename Dtype>
void HDMLLossUpperBoundLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (true || propagate_down[0]) {
    int num = bottom[0]->num();
    int nb = bottom[0]->channels();
    for (int k=0; k<num; ++k) {
      Dtype* qry_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(k);
      Dtype* pos_diff = bottom[1]->mutable_cpu_diff() + bottom[1]->offset(k);
      Dtype* neg_diff = bottom[2]->mutable_cpu_diff() + bottom[2]->offset(k);

      const int * p_h_qry = this->h_qry_.cpu_data() + this->h_qry_.offset(k);
      const int * p_h_pos = this->h_pos_.cpu_data() + this->h_pos_.offset(k);
      const int * p_h_neg = this->h_neg_.cpu_data() + this->h_neg_.offset(k);

      const int * p_g_qry = this->g_qry_.cpu_data() + this->g_qry_.offset(k);
      const int * p_g_pos = this->g_pos_.cpu_data() + this->g_pos_.offset(k);
      const int * p_g_neg = this->g_neg_.cpu_data() + this->g_neg_.offset(k);
      for (int i=0; i<nb; ++i) {
        qry_diff[i] = + p_g_qry[i] - p_h_qry[i];
        pos_diff[i] = + p_g_pos[i] - p_h_pos[i];
        neg_diff[i] = + p_g_neg[i] - p_h_neg[i];
        // LOG_IF(INFO, qry_diff[i]) << k << '\t' << i << '\t' << qry_diff[i];
        // LOG_IF(INFO, pos_diff[i]) << k << '\t' << i << '\t' << pos_diff[i];
        // LOG_IF(INFO, neg_diff[i]) << k << '\t' << i << '\t' << neg_diff[i];
      }
    }
  }
}

INSTANTIATE_CLASS(HDMLLossUpperBoundLayer);
REGISTER_LAYER_CLASS(HDMLLossUpperBound);

}  // namespace caffe
