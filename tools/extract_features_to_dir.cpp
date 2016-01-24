#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using ::boost::filesystem::path;
using std::string;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

inline void MakeFeatDir(const string & root_dir, const string & featname) {
  LOG(INFO)<< "Creating dir for feature: " << featname;
  path feat_dir = path(root_dir) / featname;
  bool done = boost::filesystem::create_directory(feat_dir);
  if ( !done ) {
    LOG(FATAL) << "Failed to create a feature directory.";
  }
}


inline void SaveMeta(const string & root_dir, const string & featname,
    const shared_ptr<Blob<float> > & feature_blob, int num_mini_batches) {
  int batch_size = feature_blob->num();
  int channels = feature_blob->channels();
  int height = feature_blob->height();
  int width = feature_blob->width();
  path meta_path = path(root_dir) / featname / "meta";
  std::ofstream outf(meta_path.c_str(), std::ofstream::out);
  outf << "float32" << std::endl;
  outf << batch_size << '\t';
  outf << channels << '\t';
  outf << height << '\t';
  outf << width << '\n';
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    outf << batch_index << std::endl;
  }
}

inline void SaveMeta(const string & root_dir, const string & featname,
    const shared_ptr<Blob<double> > & feature_blob) {
  int batch_size = feature_blob->num();
  int channels = feature_blob->channels();
  int height = feature_blob->height();
  int width = feature_blob->width();
  path meta_path = path(root_dir) / featname / "meta";
  std::ofstream outf(meta_path.c_str(), std::ofstream::out);
  outf << "float64" << std::endl;
  outf << batch_size << '\t';
  outf << channels << '\t';
  outf << height << '\t';
  outf << width << '\n';
}

template<typename Dtype>
inline void DumpFeat(const string & root_dir, const string & featname,
    const shared_ptr<Blob<Dtype> > & feature_blob, int batch_index) {
  std::stringstream ssidx;
  ssidx << batch_index;
  path feat_path = path(root_dir) / featname / ssidx.str();
  std::ofstream outf(feat_path.c_str(), std::ofstream::out|std::ofstream::binary);
  const Dtype* feature_blob_data = feature_blob->cpu_data();
  outf.write((char *)feature_blob_data, feature_blob->count() * sizeof(Dtype));
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dir  num_mini_batches "
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names separated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  std::string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string root_dir(argv[++arg_pos]);

  size_t num_features = blob_names.size();
  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  for (size_t i = 0; i < num_features; i++) {
    MakeFeatDir(root_dir, blob_names[i]);
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
  std::vector<Blob<float>*> input_vec;
  std::vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    if (batch_index % 100 == 0) {
      LOG(ERROR) << "\t" << batch_index << "/" << num_mini_batches;
    }
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < num_features; ++i) {
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          ->blob_by_name(blob_names[i]);
      DumpFeat(root_dir, blob_names[i], feature_blob, batch_index);
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  LOG(INFO) << "\t" << num_mini_batches << "/" << num_mini_batches;
  // write the meta data
  for (int i = 0; i < num_features; ++i) {
    const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
      ->blob_by_name(blob_names[i]);
    SaveMeta(root_dir, blob_names[i], feature_blob, num_mini_batches);
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
