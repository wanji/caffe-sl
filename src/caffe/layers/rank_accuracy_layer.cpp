#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rank_accuracy_layer.hpp"

namespace caffe {

template <typename Dtype>
void RankAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  vector<int> acc_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(acc_shape);
}

template <typename Dtype>
void RankAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* pos_sim = bottom[0]->cpu_data();
  const Dtype* neg_sim = bottom[1]->cpu_data();
  int count = bottom[0]->count();

  Dtype* accuracy = top[0]->mutable_cpu_data();
  accuracy[0] = 0;
  for (int i=0; i<count; ++i) {
    accuracy[0] += (pos_sim[i] > neg_sim[i] ? 1 : 0);
  }
  accuracy[0] /= count;
}

INSTANTIATE_CLASS(RankAccuracyLayer);
REGISTER_LAYER_CLASS(RankAccuracy);

}  // namespace caffe
