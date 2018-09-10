#include <vector>
#include <iostream>
#include <time.h>
#include <stdio.h>

#include "caffe/layers/dropdepth_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void DropdepthLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	threshold_ = this->layer_param_.dropdepth_param().dropdepth_ratio();
	DCHECK(threshold_ > 0.);
	int_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);    
}

template <typename Dtype>
void DropdepthLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::Reshape(bottom, top);
	rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DropdepthLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	unsigned int* mask = rand_vec_.mutable_cpu_data();
	unsigned int* a = new unsigned int;
	if (this->phase_ == TRAIN){
		caffe_rng_bernoulli(1, 1. - threshold_, a);      // a is 0 or 1
		for (int i = 0; i < count; ++i){
			mask[i] = *a;                                // put a in mask
			top_data[i] = bottom_data[i] * mask[i];
		}
	}
	else{
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}
	delete[] a;
}

template <typename Dtype>
void DropdepthLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom){
	if (propagate_down[0]){
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (this->phase_ == TRAIN){
			const unsigned int* mask = rand_vec_.cpu_data();
			const int count = bottom[0]->count();
			for (int i = 0; i < count; ++i) {
				bottom_diff[i] = top_diff[i] * mask[i];
			}
		}
		else{
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(DropdepthLayer);
#endif

INSTANTIATE_CLASS(DropdepthLayer);
REGISTER_LAYER_CLASS(Dropdepth);

}  // namespace caffe
