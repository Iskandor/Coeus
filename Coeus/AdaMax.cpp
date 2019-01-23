#include "AdaMax.h"
#include "AdaMaxRule.h"

using namespace Coeus;

AdaMax::AdaMax(NeuralNetwork* p_network) : GradientAlgorithm(p_network) {
}

AdaMax::~AdaMax()
= default;

void AdaMax::init(ICostFunction* p_cost_function, const double p_alpha, const double p_beta1, const double p_beta2, const double p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new AdaMaxRule(_network_gradient, p_alpha, p_beta1, p_beta2, p_epsilon));
}

