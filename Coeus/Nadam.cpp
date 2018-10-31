#include "Nadam.h"
#include "NadamRule.h"

using namespace Coeus;

Nadam::Nadam(NeuralNetwork * p_network) : BaseGradientAlgorithm(p_network)
{
}

Nadam::~Nadam()
= default;

void Nadam::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	BaseGradientAlgorithm::init(p_cost_function, new NadamRule(_network_gradient, p_alpha, p_beta1, p_beta2, p_epsilon));
}