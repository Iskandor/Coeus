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

Tensor* LinearActivation::backward(Tensor* p_input)
{
	float* arr = Tensor::alloc_arr(_output->size());
	int* shape = Tensor::copy_shape(_output->rank(), _output->shape());

	float* y = &arr[0];

	for (int i = 0; i < _output->size(); i++) {
		*y++ = 1.f;
	}

	TensorOperator::instance().vv_ewprod(arr, p_input->arr(), arr, _output->size());

	return new Tensor(_output->rank(), shape, arr);
}
