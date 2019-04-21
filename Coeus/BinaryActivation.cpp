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
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for(int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1.f : 0.f;
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor BinaryActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1.f : 0.f;
	}

	return Tensor(p_input.rank(), shape, arr);
}

float BinaryActivation::activate(const float p_value)
{
	return p_value > 0 ? 1.f : 0.f;
}
