#include "SoftplusActivation.h"
#include <cmath>

using namespace Coeus;

SoftplusActivation::SoftplusActivation(): IActivationFunction(SOFTPLUS) {
}


SoftplusActivation::~SoftplusActivation()
{
}

Tensor SoftplusActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = log(1 + exp((*x++)));
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor SoftplusActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = 1 / (1 + exp(-(*x++)));
	}

	return Tensor(p_input.rank(), shape, arr);
}

float SoftplusActivation::activate(const float p_value)
{
	return log(1 + exp(p_value));
}