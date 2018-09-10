#ifndef CAFFE_DROPDEPTH_LAYER_HPP_
#define CAFFE_DROPDEPTH_LAYER_HPP_

#include <vector>
#include <iostream>
#include <time.h>
#include <stdio.h>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
template <typename Dtype>
class DropdepthLayer : public NeuronLayer < Dtype > {
public:
	explicit DropdepthLayer(const LayerParameter& param)
		: NeuronLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline const char* type() const { return "Dropdepth"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<unsigned int> rand_vec_;
	Dtype threshold_;
	unsigned int int_thres_;
};

}  // namespace caffe

#endif
