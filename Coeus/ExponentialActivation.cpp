#include "ExponentialActivation.h"
#include <cmath>
#include <cstring>
#include "TensorOperator.h"

using namespace Coeus;

ExponentialActivation::ExponentialActivation(const int p_k): IActivationFunction(EXP), _k(p_k) {
}

ExponentialActivation::~ExponentialActivation()
{
}

Tensor* ExponentialActivation::backward(Tensor* p_input)
{
	float* arr = Tensor::alloc_arr(_output->size());
	int* shape = Tensor::copy_shape(_output->rank(), _output->shape());
	float* y = &arr[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < _input->size(); i++) {
		(*y++) = -exp(-_k * *x++);
	}

	TensorOperator::instance().vv_ewprod(arr, p_input->arr(), arr, _output->size());

	return new Tensor(_output->rank(), shape, arr);
}

Tensor* ExponentialActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);

	float* y = &_output->arr()[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < p_input->size(); i++) {
		*y++ = exp(-_k * *x++);
	}

	return _output;
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
