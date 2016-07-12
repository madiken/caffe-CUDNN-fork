#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {
template <typename Dtype>
void  generateMasks(int sz_h, int sz_w, int patch_h, int patch_w, Dtype* dest){
    int num_h = (int) (sz_h / patch_h);
    int num_w = (int) (sz_w / patch_w);
    
    if (sz_h % patch_h > 0)
        num_h += 1;
    if (sz_w % patch_w > 0)
        num_w += 1;

    caffe_set(num_h*num_w*sz_h*sz_w, (Dtype)0.0, dest);

    for (int ii = 0; ii < sz_h; ii++){
        for (int jj = 0; jj < sz_w; jj++){
            int i = (int)ii/patch_h;
            int j = (int)jj/patch_w;
            
            int mask_num = i*num_w + j;
            dest[mask_num * sz_h * sz_w  + ii*sz_w + jj] = 1;
        }
    }
}


template <typename Dtype>
void BilinearV2Layer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());

  CHECK_EQ(bottom[0]->width()*bottom[0]->height(),bottom[1]->width()*bottom[1]->height());

  int patch_h = this->layer_param_.bilinear_v2_param().patch_h();
  int patch_w = this->layer_param_.bilinear_v2_param().patch_w();
  num_h = (int) (bottom[0]->height() / patch_h);
  num_w = (int) (bottom[0]->width() / patch_w);
    
  if (bottom[0]->height() % patch_h > 0)
    num_h += 1;
  if (bottom[0]->width()% patch_w > 0)
    num_w += 1;

  poolingFieldsNum = num_h * num_w;

  masked_buffer1.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  masked_buffer2.Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());


  dlda_buffer.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  dldb_buffer.Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
  
  mask_buffer.Reshape(1, poolingFieldsNum, bottom[0]->height(), bottom[0]->width());

  generateMasks(bottom[0]->height(), bottom[0]->width(), patch_h, patch_w, mask_buffer.mutable_cpu_data());

  transposeBuffer.Reshape(1, poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels(), 1, 1);
  identityMatrix1.Reshape(1, num_w*num_h*num_w*num_h, 1, 1);
  caffe_set(identityMatrix1.count(), (Dtype)0.0, identityMatrix1.mutable_cpu_data());
   
  for(int i = 0; i <  num_w*num_h; i++ )
      identityMatrix1.mutable_cpu_data()[i * num_w*num_h + i] = (Dtype)1.0;
  

   
  identityMatrix2.Reshape(1, bottom[0]->channels()*bottom[1]->channels()* bottom[0]->channels()*bottom[1]->channels(), 1, 1);
  caffe_set(identityMatrix2.count(), (Dtype)0.0, identityMatrix2.mutable_cpu_data());
  
  for(int i = 0; i <  bottom[0]->channels()*bottom[1]->channels(); i++ )
     identityMatrix2.mutable_cpu_data()[i *   bottom[0]->channels()*bottom[1]->channels() + i] = (Dtype)1.0;
      
}
     
template <typename Dtype>
void multiplyAllChannelsByMask(const Dtype* blob, const Dtype*  mask_blob, int mask_num, Dtype* blob_result, int sz, int blob_channels){
  int data_offset = 0;
  int mask_offset = mask_num * sz;

    for(int j = 0; j < blob_channels; j++){
      data_offset = j * sz;      
      caffe_mul(sz, blob + data_offset, mask_blob + mask_offset, blob_result + data_offset);
    }
}
    
template <typename Dtype>
void BilinearV2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 // outer - positions, inner - channels
 
  top[0]->Reshape(bottom[0]->num(),  poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels(), 1, 1);   
 //  top[0]->Reshape(bottom[0]->num(),   bottom[0]->channels() * bottom[1]->channels(), num_h, num_w);   
}

template <typename Dtype>
void BilinearV2Layer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  

  for (int n = 0; n < bottom[0]->num(); n++){
 
    for (int i = 0; i < poolingFieldsNum; i++){
       multiplyAllChannelsByMask(bottom[0]->cpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, mask_buffer.cpu_data(), i, masked_buffer1.mutable_cpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

       multiplyAllChannelsByMask(bottom[1]->cpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, mask_buffer.cpu_data(), i, masked_buffer2.mutable_cpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

       //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),(Dtype)1., masked_buffer1.cpu_data(), masked_buffer2.cpu_data(), (Dtype)0., top[0]->mutable_cpu_data() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels());
        
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),(Dtype)1., masked_buffer1.cpu_data(), masked_buffer2.cpu_data(), (Dtype)0., transposeBuffer.mutable_cpu_data() + i * bottom[0]->channels() * bottom[1]->channels());
        
    }

      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,  bottom[0]->channels()*bottom[1]->channels(), num_w*num_h, num_w*num_h, (Dtype)1., transposeBuffer.mutable_cpu_data(), identityMatrix1.cpu_data(), (Dtype)0.,top[0]->mutable_cpu_data() + n*poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels());
    
  }
}
template <typename Dtype>
void BilinearV2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_set(bottom[0]->num()*bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->num()*bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), Dtype(0.0), bottom[1]->mutable_cpu_diff());


  for (int n = 0; n < bottom[0]->num(); n++){
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_w*num_h, bottom[0]->channels()*bottom[1]->channels(),   bottom[0]->channels()*bottom[1]->channels(), (Dtype)1., top[0]->mutable_cpu_diff() + n*poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels() , identityMatrix2.cpu_data(), (Dtype)0.,transposeBuffer.mutable_cpu_diff());

    for(int i = 0; i < poolingFieldsNum; i++){
      if (propagate_down[0]) {
        
        multiplyAllChannelsByMask(bottom[1]->cpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, mask_buffer.cpu_data(), i, masked_buffer2.mutable_cpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

       // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width()*bottom[0]->height(), bottom[1]->channels(),(Dtype)1., top[0]->cpu_diff() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer2.cpu_data(), (Dtype)0., dlda_buffer.mutable_cpu_diff());
	 caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width()*bottom[0]->height(), bottom[1]->channels(),(Dtype)1., transposeBuffer.cpu_diff() + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer2.cpu_data(), (Dtype)0., dlda_buffer.mutable_cpu_diff());
	
	
	multiplyAllChannelsByMask(dlda_buffer.cpu_diff(), mask_buffer.cpu_data(), i,dlda_buffer.mutable_cpu_diff(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

        caffe_add(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), dlda_buffer.cpu_diff(), bottom[0]->cpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, bottom[0]->mutable_cpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n);

      }
	
      if (propagate_down[1]) {

         multiplyAllChannelsByMask(bottom[0]->cpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, mask_buffer.cpu_data(), i, masked_buffer1.mutable_cpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());
        
       // caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), bottom[0]->channels(),(Dtype)1., top[0]->cpu_diff() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer1.cpu_data(), (Dtype)0., dldb_buffer.mutable_cpu_diff());
         caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), bottom[0]->channels(),(Dtype)1., transposeBuffer.cpu_diff() + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer1.cpu_data(), (Dtype)0., dldb_buffer.mutable_cpu_diff());


	multiplyAllChannelsByMask(dldb_buffer.cpu_diff(), mask_buffer.cpu_data(), i,dldb_buffer.mutable_cpu_diff(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

        caffe_add(bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), dldb_buffer.cpu_diff(), bottom[1]->cpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, bottom[1]->mutable_cpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n);

      }
    }
  }

}


#ifdef CPU_ONLY
STUB_cpu(BilinearV2Layer);
#endif

INSTANTIATE_CLASS(BilinearV2Layer);
REGISTER_LAYER_CLASS(BilinearV2);
}  // namespace caffe

