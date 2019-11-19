#include "SARSA.h"
#include "RuleFactory.h"

using namespace Coeus;

SARSA::SARSA(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda):
	_alpha(p_alpha), _gamma(p_gamma)
{
	_network = p_network;
	_network_gradient = new NetworkGradient(p_network);
	_update_rule = RuleFactory::create_rule(p_grad_rule, p_network, p_alpha);
}

SARSA::~SARSA()
{
	delete _network_gradient;
	delete _update_rule;
}

float SARSA::train(Tensor* p_state0, const int p_action0, Tensor* p_state1, const int p_action1, const float p_reward, const bool p_final) const
{
	_network->activate(p_state1);
	const float Qs1a1 = _network->get_output()->at(p_action1);
	_network->activate(p_state0);
	const float Qs0a0 = _network->get_output()->at(p_action0);
	const float delta = p_final ? p_reward : p_reward + _gamma * Qs1a1 - Qs0a0;
	Tensor loss({ _network->get_output_dim() }, Tensor::ZERO);
	loss[p_action0] = Qs0a0 - delta;

	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient(), _alpha);
	_network->update(_update_rule->get_update());

	return Qs0a0;
}