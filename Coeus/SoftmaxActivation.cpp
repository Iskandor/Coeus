#include "SoftmaxActivation.h"
#include <cmath>

using namespace Coeus;

SoftmaxActivation::SoftmaxActivation(): IActivationFunction(SOFTMAX) {
}


SoftmaxActivation::~SoftmaxActivation()
{
}

Tensor SoftmaxActivation::activate(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());

	float esum = 0;
	const int max = p_input.max_value_index();

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = exp(p_input[i] - p_input[max]);
		esum += arr[i];
	}

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = arr[i] / esum;
	}

	return Tensor(p_input.rank(), shape, arr);
}

Tensor SoftmaxActivation::derivative(Tensor& p_input) {
	float* arr = Tensor::alloc_arr(p_input.size() * p_input.size());

	const Tensor activation = activate(p_input);

	for (int i = 0; i < p_input.size(); i++) {
		for (int j = 0; j < p_input.size(); j++) {
			arr[i * p_input.size() + j] = activation[i] * (Tensor::kronecker_delta(i, j) - activation[j]);
		}
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}

float SoftmaxActivation::activate(float p_value)
{
	return 0;
}