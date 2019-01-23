#include "RMSProp.h"
#include "RMSPropRule.h"

using namespace Coeus;

RMSProp::RMSProp(NeuralNetwork* p_network) : GradientAlgorithm(p_network)
{
}


RMSProp::~RMSProp()
= default;

void RMSProp::init(ICostFunction* p_cost_function, const double p_alpha, const double p_decay, const double p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new RMSPropRule(_network_gradient, p_alpha, p_decay, p_epsilon));
}