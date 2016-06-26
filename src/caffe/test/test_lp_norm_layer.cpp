#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lp_norm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class L2NormLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  L2NormLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~L2NormLayerTest() { delete blob_bottom_; delete blob_top_; }

  Dtype epsilon_;

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(L2NormLayerTest, TestDtypesAndDevices);

TYPED_TEST(L2NormLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  L2NormLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
  EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_->height());
  EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_->width());
  const int dim = this->blob_bottom_->count() / this->blob_bottom_->num();
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    const Dtype* p_btm = this->blob_bottom_->cpu_data() + this->blob_bottom_->offset(i);
    const Dtype* p_top = this->blob_top_->cpu_data() + this->blob_top_->offset(i);
    Dtype btm_norm = 0.0;
    Dtype top_norm = 0.0;
    for (int j=0; j<dim; ++j) {
      btm_norm += p_btm[j] * p_btm[j];
      top_norm += p_top[j] * p_top[j];
    }
    EXPECT_NEAR(top_norm, 1.0, this->epsilon_);
    for (int j=0; j<dim; ++j) {
      EXPECT_NEAR(p_btm[j], sqrt(btm_norm) * p_top[j], this->epsilon_);
    }
    // EXPECT_EQ(norm, 1.0);
  }
}

TYPED_TEST(L2NormLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  L2NormLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(L2NormLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  L2NormLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

}  // namespace caffe
