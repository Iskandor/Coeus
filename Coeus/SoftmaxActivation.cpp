#include "SoftmaxActivation.h"
#include <cmath>
#include "FLAB.h"

using namespace Coeus;

SoftmaxActivation::SoftmaxActivation()
{
}


SoftmaxActivation::~SoftmaxActivation()
{
}

Tensor SoftmaxActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());
	double esum = 0;
	const int max = p_input.max_value_index();

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = exp(p_input[i] - p_input[max]);
		esum += arr[i];
	}

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = arr[i] / esum;
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor SoftmaxActivation::deriv(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		for (int j = 0; j < p_input.size(); j++) {
			arr[i * p_input.size() + j] = p_input[i] * (kronecker_delta(i, j) - p_input[j]);
		}
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}
