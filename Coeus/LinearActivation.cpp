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

Tensor LinearActivation::deriv(Tensor& p_input) { 
	return Tensor::Ones({ p_input.size() }).diag();
}
