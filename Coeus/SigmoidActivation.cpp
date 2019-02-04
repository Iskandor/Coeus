#include "SigmoidActivation.h"
#include <cmath>
#include <cstring>

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

Tensor SigmoidActivation::derivative(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	const Tensor activation = activate(p_input);

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = activation[i] * (1 - activation[i]);
	}

	return Tensor({ p_input.size() }, arr);
}

double SigmoidActivation::activate(const double p_value)
{
	return 1 / (1 + exp(-p_value));
}
