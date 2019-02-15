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
	return Tensor::Ones({ p_input.size() });
}

float LinearActivation::activate(const float p_value)
{
	return p_value;
}
