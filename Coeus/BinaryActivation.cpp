#include "BinaryActivation.h"

using namespace Coeus;

BinaryActivation::BinaryActivation() : IActivationFunction(BINARY)
{
}


BinaryActivation::~BinaryActivation()
{
}

Tensor BinaryActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for(int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1.f : 0.f;
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor BinaryActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1.f : 0.f;
	}

	return Tensor({ p_input.size() }, arr);
}

float BinaryActivation::activate(const float p_value)
{
	return p_value > 0 ? 1.f : 0.f;
}
