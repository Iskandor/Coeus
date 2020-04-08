#include "SARSA.h"


SARSA::SARSA(neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma) :
	_critic(p_critic),
	_critic_optimizer(p_critic_optimizer),
	_gamma(p_gamma)
{
}

SARSA::~SARSA()
= default;

tensor& SARSA::get_action(tensor* p_state)
{
	tensor& q_values = _critic->forward(p_state);
	_action = tensor::zero_like(q_values);
	_action[q_values.max_index()[0]] = 1.f;

	return _action;
}

void SARSA::train(tensor* p_state, tensor* p_action, tensor* p_next_state, tensor* p_next_action, float p_reward, bool p_final)
{
	_critic->backward(critic_loss_function(p_state, p_action, p_next_state, p_next_action, p_reward, p_final));
	_critic_optimizer->update();
}

tensor& SARSA::critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, tensor* p_next_action, float p_reward, bool p_final)
{
	tensor& q_next_values = _critic->forward(p_next_state);
	const int a1_index = p_action->max_index()[0];
	const float Qs1a1 = q_next_values[a1_index];

	tensor& q_values = _critic->forward(p_state);
	const int a0_index = p_action->max_index()[0];
	const float Qs0a0 = q_values[a0_index];

	_critic_loss = tensor::zero_like(q_values);
	if (p_final)
	{
		_critic_loss[a0_index] = Qs0a0 - p_reward;
	}
	else
	{
		_critic_loss[a0_index] = Qs0a0 - (p_reward + _gamma * Qs1a1);
	}

	return _critic_loss;
}
