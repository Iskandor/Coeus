#include "TanhActivation.h"
#include <cmath>

using namespace Coeus;

TanhActivation::TanhActivation()
{
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
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = 1 - pow(p_input[i], 2);
	}

	return Tensor({ p_input.size() }, arr).diag();
}