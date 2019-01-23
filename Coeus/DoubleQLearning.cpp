#include "DoubleQLearning.h"
#include "RandomGenerator.h"

using namespace Coeus;

DoubleQLearning::DoubleQLearning(NeuralNetwork* p_network_a, GradientAlgorithm* p_gradient_algorithm_a, NeuralNetwork* p_network_b, GradientAlgorithm* p_gradient_algorithm_b, const double p_gamma) {
	_network_a = p_network_a;
	_network_b = p_network_b;
	_gradient_algorithm_a = p_gradient_algorithm_a;
	_gradient_algorithm_b = p_gradient_algorithm_b;
	_gamma = p_gamma;

	_target = Tensor::Zero({ _network_a->get_output()->size() });
}

DoubleQLearning::~DoubleQLearning()
{
}

double DoubleQLearning::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const double p_reward) {

	double error;

	if (RandomGenerator::getInstance().random() > 0.5) {
		const double maxQs1a = calc_max_qa(p_state1, _network_a);
		_network_a->activate(p_state0);
		_target.override(_network_a->get_output());
		_target[p_action0] = p_reward + _gamma * maxQs1a;
		error = _gradient_algorithm_a->train(p_state0, &_target);
	}
	else {
		const double maxQs1a = calc_max_qa(p_state1, _network_b);
		_network_b->activate(p_state0);
		_target.override(_network_b->get_output());
		_target[p_action0] = p_reward + _gamma * maxQs1a;
		error = _gradient_algorithm_b->train(p_state0, &_target);
	}

	return error;
}

double DoubleQLearning::calc_max_qa(Tensor* p_state, NeuralNetwork* p_network) const {
	p_network->activate(p_state);
	const int maxQa = p_network->get_output()->max_value_index();

	return p_network->get_output()->at(maxQa);
}
