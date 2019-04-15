#include "TanhActivation.h"
#include <cmath>
#include <cstring>

using namespace Coeus;

TanhActivation::TanhActivation(): IActivationFunction(TANH) {
}


TanhActivation::~TanhActivation()
{
}

Tensor TanhActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = tanh((*x++));
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor TanhActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = 1 - tanh(*x) * tanh(*x);
		x++;
	}

	return Tensor(p_input.rank(), shape, arr);
}

float TanhActivation::activate(const float p_value)
{
	return tanh(p_value);
}
