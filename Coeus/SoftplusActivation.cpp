#include "SoftplusActivation.h"
#include <cmath>
#include <cstring>

using namespace Coeus;

SoftplusActivation::SoftplusActivation(): IActivationFunction(SOFTPLUS) {
}


SoftplusActivation::~SoftplusActivation()
{
}

Tensor SoftplusActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = log(1 + exp((*x++)));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor SoftplusActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = 1 / (1 + exp(-(*x++)));
	}

	return Tensor({ p_input.size() }, arr);
}

double SoftplusActivation::activate(const double p_value)
{
	return log(1 + exp(p_value));
}
