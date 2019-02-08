#include "ReluActivation.h"
#include <algorithm>

using namespace Coeus;

ReluActivation::ReluActivation(): IActivationFunction(RELU) {
}


ReluActivation::~ReluActivation()
{
}

Tensor ReluActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = std::max(0., *x++);
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor ReluActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = *x++ > 0 ? 1 : 0;
	}

	return Tensor({ p_input.size() }, arr);
}

double ReluActivation::activate(const double p_value)
{
	return std::max(0., p_value);
}
