#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <math.h> 
namespace caffe {


template <typename Dtype>
void multiplyAllChannelsByMaskGpu(const Dtype* blob, const Dtype*  mask_blob, int mask_num, Dtype* blob_result, int sz, int blob_channels){
  int data_offset = 0;
  int mask_offset = mask_num * sz;

    for(int j = 0; j < blob_channels; j++){
      data_offset = j * sz;      
      caffe_gpu_mul(sz, blob + data_offset, mask_blob + mask_offset, blob_result + data_offset);
    }
}


template <typename Dtype>
void BilinearV2Layer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {  

  for (int n = 0; n < bottom[0]->num(); n++){
    for (int i = 0; i < poolingFieldsNum; i++){
       multiplyAllChannelsByMaskGpu(bottom[0]->gpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, mask_buffer.gpu_data(), i, masked_buffer1.mutable_gpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

       multiplyAllChannelsByMaskGpu(bottom[1]->gpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, mask_buffer.gpu_data(), i, masked_buffer2.mutable_gpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

       //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),(Dtype)1., masked_buffer1.gpu_data(), masked_buffer2.gpu_data(), (Dtype)0., top[0]->mutable_gpu_data() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, bottom[0]->channels(), bottom[1]->channels(), bottom[0]->height() * bottom[0]->width(),(Dtype)1., masked_buffer1.gpu_data(), masked_buffer2.gpu_data(), (Dtype)0., transposeBuffer.mutable_gpu_data()   + i * bottom[0]->channels() * bottom[1]->channels());   

    }
 
   caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[0]->channels()*bottom[1]->channels(), num_w*num_h, num_w*num_h, (Dtype)1., transposeBuffer.mutable_gpu_data() , identityMatrix1.gpu_data(), (Dtype)0.,top[0]->mutable_gpu_data() + n*poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels());
  

  }
}

template <typename Dtype>
void BilinearV2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_set(bottom[0]->num()*bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), Dtype(0.0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->num()*bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), Dtype(0.0), bottom[1]->mutable_gpu_diff());


  for (int n = 0; n < bottom[0]->num(); n++){
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_w*num_h, bottom[0]->channels()*bottom[1]->channels(),   bottom[0]->channels()*bottom[1]->channels(), (Dtype)1., top[0]->mutable_gpu_diff() + n*poolingFieldsNum * bottom[0]->channels() * bottom[1]->channels() , identityMatrix2.gpu_data(), (Dtype)0.,transposeBuffer.mutable_gpu_diff());

    for(int i = 0; i < poolingFieldsNum; i++){
      if (propagate_down[0]) {
        
        multiplyAllChannelsByMaskGpu(bottom[1]->gpu_data() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, mask_buffer.gpu_data(), i, masked_buffer2.mutable_gpu_data(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

       // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width()*bottom[0]->height(), bottom[1]->channels(),(Dtype)1., top[0]->gpu_diff() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer2.gpu_data(), (Dtype)0., dlda_buffer.mutable_gpu_diff());
	 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[0]->channels(), bottom[0]->width()*bottom[0]->height(), bottom[1]->channels(),(Dtype)1., transposeBuffer.gpu_diff() + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer2.cpu_data(), (Dtype)0., dlda_buffer.mutable_gpu_diff());
	
	
	multiplyAllChannelsByMaskGpu(dlda_buffer.gpu_diff(), mask_buffer.gpu_data(), i,dlda_buffer.mutable_gpu_diff(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());

        caffe_gpu_add(bottom[0]->channels()*bottom[0]->height()*bottom[0]->width(), dlda_buffer.gpu_diff(), bottom[0]->gpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, bottom[0]->mutable_gpu_diff() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n);

      }
	
      if (propagate_down[1]) {

         multiplyAllChannelsByMaskGpu(bottom[0]->gpu_data() + bottom[0]->channels() * bottom[0]->height() * bottom[0]->width() * n, mask_buffer.gpu_data(), i, masked_buffer1.mutable_gpu_data(), bottom[0]->height()*bottom[0]->width(), bottom[0]->channels());
        
       // caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), bottom[0]->channels(),(Dtype)1., top[0]->gpu_diff() + n * top[0]->channels()  + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer1.gpu_data(), (Dtype)0., dldb_buffer.mutable_gpu_diff());
         caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, bottom[1]->channels(), bottom[1]->width()*bottom[1]->height(), bottom[0]->channels(),(Dtype)1., transposeBuffer.gpu_diff() + i * bottom[0]->channels() * bottom[1]->channels(), masked_buffer1.gpu_data(), (Dtype)0., dldb_buffer.mutable_gpu_diff());


	multiplyAllChannelsByMaskGpu(dldb_buffer.gpu_diff(), mask_buffer.gpu_data(), i,dldb_buffer.mutable_gpu_diff(), bottom[1]->height()*bottom[1]->width(), bottom[1]->channels());

        caffe_gpu_add(bottom[1]->channels()*bottom[1]->height()*bottom[1]->width(), dldb_buffer.gpu_diff(), bottom[1]->gpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n, bottom[1]->mutable_gpu_diff() + bottom[1]->channels() * bottom[1]->height() * bottom[1]->width() * n);

      }
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(BilinearV2Layer);
}  // namespace caffe

