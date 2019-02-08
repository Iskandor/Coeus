#include "TanhActivation.h"
#include <cmath>
#include <cstring>

using namespace Coeus;

TanhActivation::TanhActivation(): IActivationFunction(TANH) {
}


TanhActivation::~TanhActivation()
{
}

Tensor TanhActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double* y = &arr[0];
	double* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = tanh((*x++));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor TanhActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	const Tensor activation = activate(p_input);
	double* y = &arr[0];
	double* x = &activation.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = 1 - pow((*x++), 2);
	}

	return Tensor({ p_input.size() }, arr);
}

double TanhActivation::activate(const double p_value)
{
	return tanh(p_value);
}
