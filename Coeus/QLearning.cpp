#include "QLearning.h"

using namespace Coeus;

QLearning::QLearning(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, const double p_gamma, const double p_alpha)
{
	_network = p_network;
	_gradient_algorithm = p_gradient_algorithm;
	_alpha = p_alpha;
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
	_target[p_action0] = _target[p_action0] + _alpha * (p_reward + _gamma * maxQs1a - _target[p_action0]);

	const double error = _gradient_algorithm->train(p_state0, &_target);

	return error;
}

double QLearning::calc_max_qa(Tensor* p_state) const {
	_network->activate(p_state);
	const int maxQa = _network->get_output()->max_value_index();

	return _network->get_output()->at(maxQa);
}


