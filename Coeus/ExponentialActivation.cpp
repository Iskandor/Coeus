#include "ExponentialActivation.h"
#include <cmath>

using namespace Coeus;

ExponentialActivation::ExponentialActivation(const int p_k): IActivationFunction(EXPONENTIAL), _k(p_k) {
}

ExponentialActivation::~ExponentialActivation()
{
}

Tensor ExponentialActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = exp(-_k * p_input[i]);
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor ExponentialActivation::deriv(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = -exp(-_k * p_input[i]);
	}

	return Tensor({ p_input.size() }, arr).diag();

}
