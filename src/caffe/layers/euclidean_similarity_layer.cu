#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/euclidean_similarity_layer.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanSimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype * sim = top[0]->mutable_cpu_data();
  const Dtype * pd = diff_.cpu_data();
  for (int i=0; i<num; ++i) {
    sim[i] = caffe_cpu_dot(dim, pd, pd);
    pd += dim;
  }
  caffe_cpu_scale(num, Dtype(-0.5), sim, sim);
}

template <typename Dtype>
void EuclideanSimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  const Dtype * pd = diff_.cpu_data();
  const Dtype * pt = top[0]->cpu_diff();
  Dtype * pa = bottom[0]->mutable_cpu_diff();
  Dtype * pb = bottom[1]->mutable_cpu_diff();

  for (int i=0; i<num; ++i) {
    caffe_cpu_scale(dim, -pt[i], pd, pa);
    caffe_cpu_scale(dim,  pt[i], pd, pb);
    pd += dim; pa += dim; pb += dim;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanSimilarityLayer);

}  // namespace caffe
