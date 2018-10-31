#include "ADAM.h"
#include "ADAMRule.h"

using namespace Coeus;


ADAM::ADAM(NeuralNetwork* p_network) : BaseGradientAlgorithm(p_network) {
}

ADAM::~ADAM() {
}

void ADAM::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, new ADAMRule(_network_gradient, p_alpha, p_beta1, p_beta2, p_epsilon));
}