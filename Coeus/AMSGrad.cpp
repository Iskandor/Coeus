#include "AMSGrad.h"
#include "AMSGradRule.h"

using namespace Coeus;

AMSGrad::AMSGrad(NeuralNetwork* p_network) : GradientAlgorithm(p_network)
{
}


AMSGrad::~AMSGrad()
= default;

void AMSGrad::init(ICostFunction* p_cost_function, const float p_alpha, const float p_beta1, const float p_beta2, const float p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new AMSGradRule(_network_gradient, p_alpha, p_beta1, p_beta2, p_epsilon));
}