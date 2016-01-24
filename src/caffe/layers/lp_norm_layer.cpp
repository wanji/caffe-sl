#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lp_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  // CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
  //   "allow in-place computation.";
}

template <typename Dtype>
void L2NormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  const int num = bottom[0]->shape(0);
  this->inv_lpnorm_ = new Dtype[num];
}

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const int num = top[0]->shape(0);
  const int dim = count / num;

  Dtype * inv_norm = this->inv_lpnorm_;

  for (int i=0; i<num; ++i) {
    const Dtype * bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(i);
    inv_norm[i] = caffe_cpu_dot(dim, bottom_data, bottom_data);
  }
  for (int i=0; i<num; ++i) {
    const Dtype * bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(i);
    Dtype * top_data = top[0]->mutable_cpu_data() + top[0]->offset(i);
    inv_norm[i] = 1.0 / std::sqrt(inv_norm[i]);
    caffe_cpu_scale(dim, inv_norm[i], bottom_data, top_data);
  }
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const int count = top[0]->count();
  const int num = top[0]->shape(0);
  const int dim = count / num;

  Dtype * inv_norm = this->inv_lpnorm_;
  for (int i=0; i<num; ++i) {
    const Dtype* top_data = top[0]->cpu_data() + top[0]->offset(i);
    Dtype* top_diff = top[0]->mutable_cpu_diff() + top[0]->offset(i);
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(i);

    Dtype temp;
    caffe_cpu_scale(dim, inv_norm[i], top_diff, top_diff);
    temp = caffe_cpu_dot(dim, top_data, top_diff);
    caffe_cpu_scale(dim, -temp, top_data, bottom_diff);
    caffe_add(dim, bottom_diff, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormLayer);
#endif

INSTANTIATE_CLASS(L2NormLayer);
REGISTER_LAYER_CLASS(L2Norm);

}  // namespace caffe
