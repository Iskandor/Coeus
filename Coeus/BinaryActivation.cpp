#include "BinaryActivation.h"
#include "TensorOperator.h"

using namespace Coeus;

BinaryActivation::BinaryActivation() : IActivationFunction(BINARY)
{
}


BinaryActivation::~BinaryActivation()
{
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

Tensor* BinaryActivation::backward(Tensor* p_input)
{
	float* arr = Tensor::alloc_arr(_output->size());
	int* shape = Tensor::copy_shape(_output->rank(), _output->shape());
	float* y = &arr[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < _input->size(); i++) {
		*y++ = *x++ > 0 ? 1.f : 0.f;
	}

	TensorOperator::instance().vv_ewprod(arr, p_input->arr(), arr, _output->size());

	return new Tensor(_output->rank(), shape, arr);
}

Tensor* BinaryActivation::forward(Tensor* p_input)
{
	IActivationFunction::forward(p_input);

	float* y = &_output->arr()[0];
	float* x = &_input->arr()[0];

	for (int i = 0; i < _output->size(); i++) {
		*y++ = *x++ > 0 ? 1.f : 0.f;
	}

	return _output;
}