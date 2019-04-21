#include "ReluActivation.h"

using namespace Coeus;

ReluActivation::ReluActivation(): IActivationFunction(RELU) {
}


ReluActivation::~ReluActivation()
= default;

Tensor ReluActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) =  Tensor::max(0, *x++);
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor ReluActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0.f ? 1.f : 0.f;
	}

	return Tensor(p_input.rank(), shape, arr);
}

float ReluActivation::activate(const float p_value)
{
	return Tensor::max(0, p_value);
}