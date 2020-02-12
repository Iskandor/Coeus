#include "QLearning.h"
#include "NetworkGradient.h"
#include "RuleFactory.h"

using namespace Coeus;

QLearning::QLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, const float p_alpha, const float p_gamma): ICritic(p_network, p_grad_rule, p_alpha),
	_gamma(p_gamma)
{
}

QLearning::~QLearning()
= default;

float QLearning::train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, const float p_reward, const bool p_final)
{
	const float maxQs1a = calc_max_qa(p_state1);
	const int action = p_action->max_value_index();

	_network->activate(p_state0);
	const float Qs0a = _network->get_output()->at(action);
	float delta = p_reward;

	if (!p_final) delta += _gamma * maxQs1a - Qs0a;

	Tensor loss({ _network->get_output_dim() }, Tensor::ZERO);
	loss[action] = Qs0a - delta;

	_network_gradient->calc_gradient(&loss);
	_update_rule->calc_update(_network_gradient->get_gradient());
	_network->update(_update_rule->get_update());

	return Qs0a;
}

float QLearning::calc_max_qa(Tensor* p_state) const
{
	_network->activate(p_state);
	const int maxQa = _network->get_output()->max_value_index();

	return _network->get_output()->at(maxQa);
}