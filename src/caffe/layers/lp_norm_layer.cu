#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/lp_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const int num = top[0]->shape(0);
  const int dim = count / num;

  Dtype * inv_norm = this->inv_lpnorm_;

  for (int i=0; i<num; ++i) {
    const Dtype * bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(i);
    caffe_gpu_dot(dim, bottom_data, bottom_data, inv_norm + i);
  }
  for (int i=0; i<num; ++i) {
    const Dtype * bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(i);
    Dtype * top_data = top[0]->mutable_gpu_data() + top[0]->offset(i);
    inv_norm[i] = 1.0 / std::sqrt(inv_norm[i]);
    caffe_gpu_scale(dim, inv_norm[i], bottom_data, top_data);
  }
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const int count = top[0]->count();
  const int num = top[0]->shape(0);
  const int dim = count / num;

  Dtype * inv_norm = this->inv_lpnorm_;
  for (int i=0; i<num; ++i) {
    const Dtype* top_data = top[0]->gpu_data() + top[0]->offset(i);
    Dtype* top_diff = top[0]->mutable_gpu_diff() + top[0]->offset(i);
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(i);

    Dtype temp;
    caffe_gpu_scale(dim, inv_norm[i], top_diff, top_diff);
    caffe_gpu_dot(dim, top_data, top_diff, &temp);
    caffe_gpu_scale(dim, -temp, top_data, bottom_diff);
    caffe_gpu_add(dim, bottom_diff, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);


}  // namespace caffe
