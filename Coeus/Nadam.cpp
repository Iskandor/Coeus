#include "Nadam.h"
#include "NadamRule.h"

using namespace Coeus;

Nadam::Nadam(NeuralNetwork * p_network) : GradientAlgorithm(p_network)
{
}

Nadam::~Nadam()
= default;

void Nadam::init(ICostFunction* p_cost_function, const float p_alpha, const float p_beta1, const float p_beta2, const float p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new NadamRule(_network_gradient, p_alpha, p_beta1, p_beta2, p_epsilon));
}