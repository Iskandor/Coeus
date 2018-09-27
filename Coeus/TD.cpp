#include "TD.h"

using namespace Coeus;

TD::TD(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, const double p_gamma, const double p_alpha) {
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_alpha = p_alpha;
	_gamma = p_gamma;

	_target = Tensor::Zero({ _network->get_output()->size() });
}

TD::~TD()
{
}

double TD::train(Tensor* p_state0, Tensor* p_state1, const double p_reward) {
	_network->activate(p_state1);
	const double Vs0 = _network->get_output()->at(0);
	_network->activate(p_state1);
	const double Vs1 = _network->get_output()->at(0);

	_target[0] = Vs0 + _alpha * (p_reward + _gamma * Vs1 - Vs0);

	const double error = _gradient_algorithm->train(p_state0, &_target);

	return error;
}
