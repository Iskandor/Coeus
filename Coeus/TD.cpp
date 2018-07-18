#include "TD.h"

using namespace Coeus;

TD::TD(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, const double p_gamma) {
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_gamma = p_gamma;

	_target = Tensor::Zero({ _network->get_output()->size() });
}

TD::~TD()
{
}

double TD::train(Tensor* p_state0, Tensor* p_state1, const double p_reward) {
	double error = 0;

	_network->activate(p_state0);
	const double Vs1 = _network->get_output()->at(0);

	_target[0] = p_reward + _gamma * Vs1;

	error = _gradient_algorithm->train(p_state0, &_target);

	return error;
}
