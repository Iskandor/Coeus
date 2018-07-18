#include "SARSA.h"

using namespace Coeus;

SARSA::SARSA(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, const double p_gamma) {
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_gamma = p_gamma;

	_target = Tensor::Zero({ _network->get_output()->size() });
}

SARSA::~SARSA() {
}

double SARSA::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const int p_action1, const double p_reward) {
	double error = 0;

	_network->activate(p_state0);
	_target.override(_network->get_output());
	_network->activate(p_state1);
	_target[p_action0] = p_reward + _gamma * _network->get_output()->at(p_action1);

	error = _gradient_algorithm->train(p_state0, &_target);

	return error;
}



