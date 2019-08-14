#include "LinearActivation.h"
#include "TensorOperator.h"

using namespace Coeus;

LinearActivation::LinearActivation(): IActivationFunction(LINEAR) {
}


LinearActivation::~LinearActivation()
= default;

Tensor* LinearActivation::forward(Tensor* p_input) {
	IActivationFunction::forward(p_input);

	_output->override(p_input);

	return _output;
}

Tensor LinearActivation::derivative(Tensor& p_input) { 
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = 1.f;
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor* LinearActivation::backward(Tensor* p_input, Tensor* p_x)
{
	IActivationFunction::backward(p_input, p_x);
	float* y = &_gradient->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		*y++ = 1.f;
	}

	TensorOperator::instance().vv_ewprod(_gradient->arr(), p_input->arr(), _gradient->arr(), _output->size());

	return _gradient;
}
