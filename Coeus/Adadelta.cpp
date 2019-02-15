#include "Adadelta.h"
#include "AdadeltaRule.h"

using namespace Coeus;

Adadelta::Adadelta(NeuralNetwork* p_network) : GradientAlgorithm(p_network) {
}


Adadelta::~Adadelta()
= default;

void Adadelta::init(ICostFunction* p_cost_function, const float p_alpha, const float p_decay, const float p_epsilon) {
	GradientAlgorithm::init(p_cost_function, new AdadeltaRule(_network_gradient, p_alpha, p_decay, p_epsilon));
}