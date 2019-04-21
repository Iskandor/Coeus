#include "ExponentialActivation.h"
#include <cmath>
#include <cstring>

using namespace Coeus;

ExponentialActivation::ExponentialActivation(const int p_k): IActivationFunction(EXP), _k(p_k) {
}

ExponentialActivation::~ExponentialActivation()
{
}

Tensor ExponentialActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = exp(-_k * *x++);
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor ExponentialActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = -exp(-_k * *x++);
	}

	return Tensor(p_input.rank(), shape, arr);
}

json ExponentialActivation::get_json()
{
	json result = IActivationFunction::get_json();

	result["k"] = _k;

	return result;
}
