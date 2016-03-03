#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/triplet_binary_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
TripletBinaryDataLayer<Dtype>::~TripletBinaryDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TripletBinaryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) && is_color)
    << "Current implementation requires "
    "none of new_height or new_width or is_color.";
  // Read the file with filenames
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  vector<string> filenames(3);
  while (infile >> filenames[0] >> filenames[1] >> filenames[2]) {
    lines_.push_back(filenames);
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleTriplets();
  }
  LOG(INFO) << "A total of " << lines_.size() << " triplets.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Load meta data
  std::ifstream metafile((root_folder + "meta").c_str());
  std::string dtype;
  metafile >> dtype;
  // int n, c, h, w;
  // metafile << n << c << h << w;
  vector<int> & top_shape = this->top_shape_;
  top_shape.resize(4);
  for (int i = 0; i < top_shape.size(); ++i) {
    metafile >> this->top_shape_[i];
  }
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  this->top_shape_[0] = batch_size * 3;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(this->top_shape_);
  }
  top[0]->Reshape(this->top_shape_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void TripletBinaryDataLayer<Dtype>::ShuffleTriplets() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void TripletBinaryDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  string root_folder = image_data_param.root_folder();

  const vector<int> & top_shape = this->top_shape_;
  // Reshape batch according to the batch_size.
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  const int batch_size = top_shape[0] / 3;
  const int count = top_shape[1] * top_shape[2] * top_shape[3];
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    for (int tri_id=0; tri_id<3; ++tri_id) {
      // get a blob
      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      int offset = batch->data_.offset(item_id + tri_id * batch_size);
      int ret = ReadBinaryBlob(root_folder + lines_[lines_id_][tri_id],
          prefetch_data + offset, count);
      read_time += timer.MicroSeconds();
      CHECK(ret == 0) << "Could not load " << lines_[lines_id_][tri_id];
    }

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleTriplets();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TripletBinaryDataLayer);
REGISTER_LAYER_CLASS(TripletBinaryData);

}  // namespace caffe
#endif  // USE_OPENCV
