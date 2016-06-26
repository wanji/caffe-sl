#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dotproduct_similarity_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class DotProductSimilarityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DotProductSimilarityLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_0_(new Blob<Dtype>()),
        blob_bottom_1_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_0_->Reshape(2, 7, 3, 3);
    blob_bottom_1_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DotProductSimilarityLayerTest() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }

  Dtype epsilon_;

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DotProductSimilarityLayerTest, TestDtypesAndDevices);

TYPED_TEST(DotProductSimilarityLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DotProductSimilarityLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int dim = this->blob_bottom_0_->count() / this->blob_bottom_0_->num();
  for (int i = 0; i < this->blob_bottom_0_->num(); ++i) {
    const Dtype* p_btm_0 = this->blob_bottom_0_->cpu_data() + this->blob_bottom_0_->offset(i);
    const Dtype* p_btm_1 = this->blob_bottom_1_->cpu_data() + this->blob_bottom_1_->offset(i);
    const Dtype* p_top = this->blob_top_->cpu_data() + this->blob_top_->offset(i);
    Dtype sim = 0.0;
    for (int j=0; j<dim; ++j) {
      sim += p_btm_0[j] * p_btm_1[j];
    }
    EXPECT_NEAR(sim, *p_top, this->epsilon_);
  }
}

TYPED_TEST(DotProductSimilarityLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DotProductSimilarityLayer<Dtype> layer(layer_param);
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

TYPED_TEST(DotProductSimilarityLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DotProductSimilarityLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_bottom_0_->num(), this->blob_bottom_1_->num());
  EXPECT_EQ(this->blob_bottom_0_->count(), this->blob_bottom_1_->count());
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

}  // namespace caffe
