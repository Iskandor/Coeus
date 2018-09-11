#include "SigmoidActivation.h"
#include <cmath>

using namespace Coeus;

SigmoidActivation::SigmoidActivation(): IActivationFunction(SIGMOID) {
}


SigmoidActivation::~SigmoidActivation()
{
}

Tensor SigmoidActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = 1 / (1 + exp(-p_input[i]));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor SigmoidActivation::deriv(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = p_input[i] * (1 - p_input[i]);;
	}

	return Tensor({ p_input.size() }, arr).diag();
}
