#include "AdaMax.h"
#include "AdaMaxRule.h"

using namespace Coeus;

AdaMax::AdaMax(NeuralNetwork* p_network) : GradientAlgorithm(p_network) {
}

AdaMax::~AdaMax()
= default;

void AdaMax::init(ICostFunction* p_cost_function, const float p_alpha, const float p_beta1, const float p_beta2, const float p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new AdaMaxRule(_network, p_alpha, p_beta1, p_beta2, p_epsilon));
}

