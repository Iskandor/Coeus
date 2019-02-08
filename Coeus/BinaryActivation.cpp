#include "BinaryActivation.h"

using namespace Coeus;

BinaryActivation::BinaryActivation() : IActivationFunction(BINARY)
{
}


BinaryActivation::~BinaryActivation()
{
}

Tensor BinaryActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for(int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1 : 0;
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor BinaryActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1 : 0;
	}

	return Tensor({ p_input.size() }, arr);
}

double BinaryActivation::activate(const double p_value)
{
	return p_value > 0 ? 1 : 0;
}
