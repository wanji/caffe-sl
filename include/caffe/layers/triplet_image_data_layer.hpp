#ifndef CAFFE_TRIPLET_IMAGE_DATA_LAYER_HPP_
#define CAFFE_TRIPLET_IMAGE_DATA_LAYER_HPP_

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
 * @brief Provides triplet data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class TripletImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TripletImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~TripletImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleTriplets();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<vector<std::string> > lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_IMAGE_DATA_LAYER_HPP_


