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

Tensor* ExponentialActivation::backward(Tensor* p_input, Tensor* p_x)
{
	IActivationFunction::backward(p_input, p_x);
	float* y = &_gradient->arr()[0];
	float* x = &_input->arr()[0];

	if (p_x != nullptr) x = &p_x->arr()[0];

	for (int i = 0; i < _input->size(); i++) {
		(*y++) = -exp(-_k * *x++);
	}

	TensorOperator::instance().vv_ewprod(_gradient->arr(), p_input->arr(), _gradient->arr(), _output->size());

	return _gradient;
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
