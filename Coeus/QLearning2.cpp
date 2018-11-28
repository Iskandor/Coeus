#include "QLearning2.h"
#include "NetworkGradient.h"
#include "QLearningRule.h"

using namespace Coeus;

QLearning2::QLearning2(NeuralNetwork* p_network, const double p_alpha, const double p_gamma, const double p_lambda):
	_gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = new QLearningRule(_network_gradient, p_alpha, p_gamma, p_lambda);
}


QLearning2::~QLearning2()
= default;

double QLearning2::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const double p_reward) const
{
	const double maxQs1a = calc_max_qa(p_state1);

	_network->activate(p_state0);
	const double Qs0a = _network->get_output()->at(p_action0);

	const double delta = p_reward + _gamma * maxQs1a - Qs0a;

	Tensor mask = Tensor::Zero({ _network->get_output()->size() });
	mask[p_action0] = 1;

	_network_gradient->calc_gradient(&mask);
	_update_rule->calc_update(_network_gradient->get_gradient(), delta);
	_network->update(_update_rule->get_update());

	return delta;
}

void QLearning2::reset_traces() const
{
	_update_rule->reset_traces();
}

double QLearning2::calc_max_qa(Tensor* p_state) const
{
	_network->activate(p_state);
	const int maxQa = _network->get_output()->max_value_index();

	return _network->get_output()->at(maxQa);
}
