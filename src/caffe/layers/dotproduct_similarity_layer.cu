#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/dotproduct_similarity_layer.hpp"

namespace caffe {

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  Dtype * sim = top[0]->mutable_cpu_data();
  const Dtype * pa = bottom[0]->gpu_data();
  const Dtype * pb = bottom[1]->gpu_data();
  for (int i=0; i<num; ++i) {
    caffe_gpu_dot(dim, pa, pb, sim + i);
    pa += dim; pb += dim;
  }
}

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int dim = count / num;
  const Dtype * top_diff = top[0]->cpu_diff();
  for (int i=0; i<num; ++i) {
    caffe_gpu_scale(dim, top_diff[i],
        bottom[0]->gpu_data() + bottom[0]->offset(i),
        bottom[1]->mutable_gpu_diff() + bottom[1]->offset(i));
    caffe_gpu_scale(dim, top_diff[i],
        bottom[1]->gpu_data() + bottom[1]->offset(i),
        bottom[0]->mutable_gpu_diff() + bottom[0]->offset(i));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DotProductSimilarityLayer);

}  // namespace caffe
