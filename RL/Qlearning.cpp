#include "Qlearning.h"

Qlearning::Qlearning(neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma) :
	_network(p_critic),
	_optimizer(p_critic_optimizer),
	_gamma(p_gamma)
{
	_delta = tensor({ 1,1 });
}

Qlearning::~Qlearning()
= default;

tensor& Qlearning::get_action(tensor* p_state)
{
	tensor& q_values = _network->forward(p_state);
	_action = tensor::zero_like(q_values);
	_action[q_values.max_index()[0]] = 1.f;

	return _action;
}

void Qlearning::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_network->backward(loss_function(p_state, p_action, p_next_state, p_reward, p_final));
	_optimizer->update();
}

tensor& Qlearning::delta()
{
	return _delta;
}

tensor& Qlearning::loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	tensor& q_next_values = _network->forward(p_next_state);
	const float max_q_value = q_next_values[q_next_values.max_index()[0]];

	tensor& q_values = _network->forward(p_state);	
	_loss = tensor::zero_like(q_values);

	const int index = p_action->max_index()[0];

	_delta[0] = q_values[index];

	if (p_final)
	{
		_loss[index] = q_values[index] - p_reward;
	}
	else
	{
		_loss[index] = q_values[index] - (p_reward + _gamma * max_q_value);
	}

	return _loss;
}
