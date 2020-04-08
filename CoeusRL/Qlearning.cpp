#include "Qlearning.h"
#include <iostream>

Qlearning::Qlearning(neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma) :
	_critic(p_critic),
	_critic_optimizer(p_critic_optimizer),
	_gamma(p_gamma)
{
}

Qlearning::~Qlearning()
= default;

tensor& Qlearning::get_action(tensor* p_state)
{
	tensor& q_values = _critic->forward(p_state);
	_action = tensor::zero_like(q_values);
	_action[q_values.max_index()[0]] = 1.f;

	return _action;
}

void Qlearning::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_critic->backward(critic_loss_function(p_state, p_action, p_next_state, p_reward, p_final));
	_critic_optimizer->update();
}

tensor& Qlearning::critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	tensor& q_next_values = _critic->forward(p_next_state);
	const float max_q_value = q_next_values[q_next_values.max_index()[0]];

	tensor& q_values = _critic->forward(p_state);
	_critic_loss = tensor::zero_like(q_values);

	const int index = p_action->max_index()[0];

	if (p_final)
	{
		_critic_loss[index] = q_values[index] - p_reward;
	}
	else
	{
		_critic_loss[index] = q_values[index] - (p_reward + _gamma * max_q_value);
	}

	return _critic_loss;
}
