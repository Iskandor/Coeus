#include "SigmoidActivation.h"
#include <cmath>
#include "TensorOperator.h"

using namespace Coeus;

SigmoidActivation::SigmoidActivation(): IActivationFunction(SIGMOID) {
}


SigmoidActivation::~SigmoidActivation()
= default;

Tensor* SigmoidActivation::backward(Tensor* p_input, Tensor* p_x)
{
	IActivationFunction::backward(p_input, p_x);
	float* y = &_gradient->arr()[0];
	float* x = &_output->arr()[0];

	if (p_x != nullptr) x = &p_x->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		(*y++) = *x * (1 - *x);
		x++;
	}

	TensorOperator::instance().vv_ewprod(_gradient->arr(), p_input->arr(), _gradient->arr(), _output->size());

	return _gradient;
}

Tensor* SigmoidActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);
	float* y = &_output->arr()[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		(*y++) = 1 / (1 + exp(-(*x++)));
	}

	return _output;
}

Tensor SigmoidActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	Tensor* activation = forward(&p_input);
	float* y = &arr[0];
	float* x = &activation->arr()[0];


	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = *x * (1 - *x);
		x++;
	}

	return Tensor(p_input.rank(), shape, arr);
}