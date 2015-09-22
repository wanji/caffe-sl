#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  // CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
  //   "allow in-place computation.";
}

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  this->lpnorm_ = caffe_cpu_dot(count, bottom[0]->cpu_data(), bottom[0]->cpu_data());
  this->lpnorm_ = std::sqrt(this->lpnorm_);
  caffe_cpu_scale(count, this->lpnorm_, bottom[0]->cpu_data(), top_data);
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    CHECK_EQ(0, 1) << "WRONG Impl !!!";
    caffe_cpu_scale(count, (Dtype)(1.0 / this->lpnorm_), top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormLayer);
#endif

INSTANTIATE_CLASS(L2NormLayer);
REGISTER_LAYER_CLASS(L2Norm);

}  // namespace caffe
