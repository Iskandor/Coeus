#include "ReluActivation.h"

using namespace Coeus;

ReluActivation::ReluActivation(): IActivationFunction(RELU) {
}


ReluActivation::~ReluActivation()
= default;

Tensor ReluActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) =  Tensor::max(0, *x++);
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor ReluActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0.f ? 1.f : 0.f;
	}

	return Tensor({ p_input.size() }, arr);
}

float ReluActivation::activate(const float p_value)
{
	return Tensor::max(0, p_value);
}