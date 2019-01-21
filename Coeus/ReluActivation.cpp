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

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = std::max(0., p_input[i]);
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor ReluActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size() * p_input.size());
	memset(arr, 0, sizeof(double) * p_input.size() * p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i*p_input.size() + i] = p_input[i] > 0 ? 1 : 0;
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}

double ReluActivation::activate(const double p_value)
{
	return std::max(0., p_value);
}
