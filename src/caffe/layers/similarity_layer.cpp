#include "caffe/layers/similarity_layer.hpp"

namespace caffe {

template <typename Dtype>
void SimilarityLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

INSTANTIATE_CLASS(SimilarityLayer);

}  // namespace caffe
