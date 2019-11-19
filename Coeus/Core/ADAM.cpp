#include "ADAM.h"
#include "ADAMRule.h"

using namespace Coeus;


ADAM::ADAM(NeuralNetwork* p_network) : GradientAlgorithm(p_network)
{
}

ADAM::~ADAM() = default;

void ADAM::init(ICostFunction* p_cost_function, const float p_alpha, const float p_beta1, const float p_beta2, const float p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new ADAMRule(_network, p_alpha, p_beta1, p_beta2, p_epsilon));
}
