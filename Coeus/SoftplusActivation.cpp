#include "SoftplusActivation.h"
#include <cmath>

using namespace Coeus;

SoftplusActivation::SoftplusActivation()
{
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

Tensor SoftplusActivation::deriv(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i] = 1 / (1 + exp(-p_input[i]));
	}

	return Tensor({ p_input.size() }, arr).diag();
}
