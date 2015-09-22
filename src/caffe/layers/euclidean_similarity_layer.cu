#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/similarity_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanSimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype * sim = top[0]->mutable_gpu_data();
  const Dtype * pd = diff_.gpu_data();
  for (int i=0; i<num; ++i) {
    caffe_gpu_dot(count, pd, pd, sim + i);
    pd += dim;
  }
  caffe_gpu_scale(count, Dtype(-1.0), sim, sim);
}

template <typename Dtype>
void EuclideanSimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  const Dtype * pd = diff_.gpu_data();
  const Dtype * pt = top[0]->gpu_diff();
  Dtype * pa = bottom[0]->mutable_gpu_diff();
  Dtype * pb = bottom[1]->mutable_gpu_diff();

  for (int i=0; i<num; ++i) {
    caffe_gpu_scale(dim, -pt[i], pd, pa);
    caffe_gpu_scale(dim,  pt[i], pd, pb);
    pd += dim; pa += dim; pb += dim;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanSimilarityLayer);

}  // namespace caffe
