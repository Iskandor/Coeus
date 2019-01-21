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

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = log(1 + exp(p_input[i]));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor SoftplusActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size() * p_input.size());
	memset(arr, 0, sizeof(double) * p_input.size() * p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i*p_input.size() + i] = 1 / (1 + exp(-p_input[i]));
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}

double SoftplusActivation::activate(const double p_value)
{
	return log(1 + exp(p_value));
}
