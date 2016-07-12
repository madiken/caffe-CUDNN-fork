#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class BilinearV2LayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BilinearV2LayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 10, 10)),
        blob_bottom_1_(new Blob<Dtype>(2, 4, 10, 10)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
   
    FillerParameter filler_param;
    filler_param.set_std(0.1);
    GaussianFiller<Dtype> filler(filler_param);
 
    filler.Fill(this->blob_bottom_0_);

    filler.Fill(this->blob_bottom_1_);

    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~BilinearV2LayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BilinearV2LayerTest, TestDtypesAndDevices);


TYPED_TEST(BilinearV2LayerTest, TestGradientTrivial) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bilinear_v2_param()->set_patch_h(5);
  layer_param.mutable_bilinear_v2_param()->set_patch_w(5);
  BilinearV2Layer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2);


  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_0_,
      this->blob_top_vec_);

}



}  // namespace caffe
