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

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = tanh(p_input[i]);
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor TanhActivation::deriv(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size() * p_input.size());
	memset(arr, 0, sizeof(double) * p_input.size() * p_input.size());

	const Tensor activation = activate(p_input);

	for (int i = 0; i < p_input.size(); i++) {
		arr[i*p_input.size()+i] = 1 - pow(activation[i], 2);
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}
