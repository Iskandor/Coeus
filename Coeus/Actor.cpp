#include "Actor.h"

using namespace Coeus;

Actor::Actor(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, const double p_gamma, const double p_beta) {
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_beta = p_beta;
	_gamma = p_gamma;

	_target = Tensor::Zero({ _network->get_output()->size() });
}

Actor::~Actor()
{
}

double Actor::train(Tensor* p_state0, const int p_action, const double p_value0, const double p_value1, const double p_reward) {
	double error = 0;

	_network->activate(p_state0);
	_target.override(_network->get_output());
	_target[p_action] = _target[p_action] + _beta * (p_reward + _gamma * p_value1 - p_value0);

	error = _gradient_algorithm->train(p_state0, &_target);

	return error;
}
