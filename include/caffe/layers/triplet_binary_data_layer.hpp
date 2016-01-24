#ifndef CAFFE_TRIPLET_BINARY_DATA_LAYER_HPP_
#define CAFFE_TRIPLET_BINARY_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides triplet data to the Net from binary files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class TripletBinaryDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TripletBinaryDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~TripletBinaryDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletBinaryData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleTriplets();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<vector<std::string> > lines_;
  vector<int> top_shape_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_BINARY_DATA_LAYER_HPP_


