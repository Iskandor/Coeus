#include "BinaryActivation.h"

using namespace Coeus;

BinaryActivation::BinaryActivation() : IActivationFunction(BINARY)
{
}


BinaryActivation::~BinaryActivation()
{
}

Tensor BinaryActivation::activate(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size());

	for(int i = 0; i < p_input.size(); i++) {
		arr[i] = p_input[i] > 0 ? 1 : 0;
	}

	return Tensor({ p_input.size() }, arr);
}

Tensor BinaryActivation::deriv(Tensor& p_input) {
	double* arr = Tensor::alloc_arr(p_input.size() * p_input.size());
	memset(arr, 0, sizeof(double) * p_input.size() * p_input.size());

	for (int i = 0; i < p_input.size(); i++) {
		arr[i*p_input.size() + i] = p_input[i] > 0 ? 1 : 0;
	}

	return Tensor({ p_input.size(), p_input.size() }, arr);
}
