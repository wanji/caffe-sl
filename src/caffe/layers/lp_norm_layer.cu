#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_dot(count, bottom[0]->gpu_data(), bottom[0]->gpu_data(), &this->lpnorm_);
  this->lpnorm_ = std::sqrt(this->lpnorm_);
  caffe_gpu_scale(count, this->lpnorm_, bottom[0]->gpu_data(), top_data);
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    CHECK_EQ(0, 1) << "WRONG Impl !!!";
    caffe_gpu_scale(count, (Dtype)(1.0 / this->lpnorm_), top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);


}  // namespace caffe
