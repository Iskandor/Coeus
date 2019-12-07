#include "RAdam.h"
#include "RAdamRule.h"
using namespace Coeus;

RADAM::RADAM(NeuralNetwork* p_network): GradientAlgorithm(p_network)
{
}

RADAM::~RADAM()
= default;

void RADAM::init(ICostFunction* p_cost_function, float p_alpha, float p_beta1, float p_beta2)
{
	GradientAlgorithm::init(p_cost_function, new RADAMRule(_network, p_alpha, p_beta1, p_beta2));
}
