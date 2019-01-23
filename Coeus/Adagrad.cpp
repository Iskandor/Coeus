#include "Adagrad.h"
#include "AdagradRule.h"

using namespace Coeus;

Adagrad::Adagrad(NeuralNetwork* p_network) : GradientAlgorithm(p_network)
{
}


Adagrad::~Adagrad()
= default;

void Adagrad::init(ICostFunction* p_cost_function, const double p_alpha, const double p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new AdagradRule(_network_gradient, p_alpha, p_epsilon));
}