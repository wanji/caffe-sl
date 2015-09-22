#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/similarity_layers.hpp"

namespace caffe {

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  Dtype * sim = top[0]->mutable_cpu_data();
  const Dtype * pa = bottom[0]->cpu_data();
  const Dtype * pb = bottom[1]->cpu_data();
  for (int i=0; i<num; ++i) {
    sim[i] = caffe_cpu_dot(count, pa, pb);
    pa += dim; pb += dim;
  }
}

template <typename Dtype>
void DotProductSimilarityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  caffe_copy(count, bottom[0]->cpu_data(), bottom[1]->mutable_cpu_data());
  caffe_copy(count, bottom[1]->cpu_data(), bottom[0]->mutable_cpu_data());
}

#ifdef CPU_ONLY
STUB_GPU(DotProductSimilarityLayer);
#endif

INSTANTIATE_CLASS(DotProductSimilarityLayer);
REGISTER_LAYER_CLASS(DotProductSimilarity);

}  // namespace caffe
