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
	float* arr = Tensor::alloc_arr(p_input.size());
	float* y = &arr[0];
	float* x = &p_input.arr()[0];

	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = 1 / (1 + exp(-(*x++)));
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor SigmoidActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	const Tensor activation = activate(p_input);
	float* y = &arr[0];
	float* x = &activation.arr()[0];


	for (int i = 0; i < p_input.size(); i++) {
		(*y++) = *x * (1 - *x);
		x++;
	}

	return Tensor({ p_input.size() }, arr);
}

float SigmoidActivation::activate(const float p_value)
{
	return 1 / (1 + exp(-p_value));
}
