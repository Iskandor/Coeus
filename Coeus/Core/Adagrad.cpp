#include "Adagrad.h"
#include "AdagradRule.h"

using namespace Coeus;

Adagrad::Adagrad(NeuralNetwork* p_network) : GradientAlgorithm(p_network)
{
}


Adagrad::~Adagrad()
= default;

void Adagrad::init(ICostFunction* p_cost_function, const float p_alpha, const float p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new AdagradRule(_network, p_alpha, p_epsilon));
}