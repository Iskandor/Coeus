#include "QLearning.h"

using namespace Coeus;

QLearning::QLearning(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, const double p_gamma)
{
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_gamma = p_gamma;

	_target = Tensor::Zero({ _network->get_output()->size() });
}

QLearning::~QLearning()
{
}

double QLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const double p_reward) {
	const double maxQs1a = calc_max_qa(p_state1);

	_network->activate(p_state0);

	_target.override(_network->get_output());
	_target[p_action0] = p_reward + _gamma * maxQs1a;

	const double error = _gradient_algorithm->train(p_state0, &_target);

	return error;
}

double QLearning::calc_max_qa(Tensor* p_state) {
	int maxQa = 0;

	_network->activate(p_state);
	for (int i = 1; i < _network->get_output()->size(); i++) {
		if (_network->get_output()->at(i) >  _network->get_output()->at(maxQa)) {
			maxQa = i;
		}
	}

	return _network->get_output()->at(maxQa);
}


