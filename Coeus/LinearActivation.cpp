#include "LinearActivation.h"

using namespace Coeus;

LinearActivation::LinearActivation(): IActivationFunction(LINEAR) {
}


LinearActivation::~LinearActivation()
{
}

Tensor LinearActivation::activate(Tensor& p_input) {
	return Tensor(p_input);
}

Tensor LinearActivation::derivative(Tensor& p_input) { 
	float* arr = Tensor::alloc_arr(p_input.size());
	int* shape = Tensor::copy_shape(p_input.rank(), p_input.shape());
	float* y = &arr[0];

	for (int i = 0; i < p_input.size(); i++) {
		*y++ = 1.f;
	}

	return Tensor(p_input.rank(), shape, arr);
}

float LinearActivation::activate(const float p_value)
{
	return p_value;
}