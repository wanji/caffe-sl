#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/similarity_layers.hpp"

namespace caffe {

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype * sim = top[0]->mutable_gpu_data();
  const Dtype * pa = bottom[0]->gpu_data();
  const Dtype * pb = bottom[1]->gpu_data();
  for (int i=0; i<num; ++i) {
    caffe_gpu_dot(count, pa, pb, sim + i);
    pa += dim; pb += dim;
  }
}

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  caffe_copy(count, bottom[0]->gpu_data(), bottom[1]->mutable_gpu_data());
  caffe_copy(count, bottom[1]->gpu_data(), bottom[0]->mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProductSimilarityLayer);

}  // namespace caffe
