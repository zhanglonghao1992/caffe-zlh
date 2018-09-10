#include <vector>
#include <iostream>
#include <time.h>
#include <stdio.h>

#include "caffe/layers/dropdepth_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropdepthForward(const int n, const Dtype* in,
    unsigned int* mask, const unsigned int threshold, Dtype* out, unsigned int* d) {
  CUDA_KERNEL_LOOP(index, n) {
    mask[index] = *d;
    out[index] = in[index] * (mask[index] < threshold);
  }
}

template <typename Dtype>
void DropdepthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    
    //unsigned int* b = new unsigned int[1];
    unsigned int b = caffe_rng_rand();
    //caffe_gpu_rng_uniform(1,b);
    //LOG(INFO)<<"b:"<<b[0];
    //LOG(INFO)<<"b:"<<b;
    //unsigned int* c;
    unsigned int* c;
    //cudaMalloc((void**)&c,1*sizeof(unsigned int));
    //cudaMemcpy(c,&b,1*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMalloc((void**)&c,1*sizeof(unsigned int));
    cudaMemcpy(c,&b,1*sizeof(unsigned int),cudaMemcpyHostToDevice);
    
    //unsigned int a;
    //cudaMemcpy(&a,c,1*sizeof(unsigned int),cudaMemcpyDeviceToHost); 
    //LOG(INFO)<<"a:"<<a;

    //LOG(INFO)<<"UINT_MAX:"<<UINT_MAX;  

    //for (int i = 0; i < count; ++i){
        //mask[i] = *c;                                
    //}
    //LOG(INFO)<<"5";
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropdepthForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, int_thres_, top_data, c);
    //LOG(INFO)<<"6";
    //delete[] b;
    cudaFree(c);
    //LOG(INFO)<<"7";  
    CUDA_POST_KERNEL_CHECK;
    //LOG(INFO)<<"8";
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropdepthBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (mask[index] < threshold);
  }
}

template <typename Dtype>
void DropdepthLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropdepthBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
         count, top_diff, mask, int_thres_, bottom_diff);
      
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropdepthLayer);

}  // namespace caffe
