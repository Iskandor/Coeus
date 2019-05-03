#include "TanhActivation.h"
#include <cmath>
#include <cstring>
#include "TensorOperator.h"

using namespace Coeus;

TanhActivation::TanhActivation(): IActivationFunction(TANH) {
}


TanhActivation::~TanhActivation()
{
}

Tensor* TanhActivation::backward(Tensor* p_input)
{
	float* arr = Tensor::alloc_arr(_output->size());
	int* shape = Tensor::copy_shape(_output->rank(), _output->shape());
	float* y = &arr[0];
	float* x = &_output->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		(*y++) = 1 - *x * *x;
		x++;
	}

	TensorOperator::instance().vv_ewprod(arr, p_input->arr(), arr, _output->size());

	return new Tensor(_output->rank(), shape, arr);
}

Tensor* TanhActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);
	float* y = &_output->arr()[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		(*y++) = tanh((*x++));
	}

	return _output;
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