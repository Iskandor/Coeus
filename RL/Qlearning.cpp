#include "Qlearning.h"

/**
 * \brief Q-learning algorithm 
 * \param p_network neural network approximating Q-function
 * \param p_optimizer network optimization algorithm
 * \param p_gamma discount factor
 */
Qlearning::Qlearning(neural_network* p_network, optimizer* p_optimizer, const float p_gamma) :
	_network(p_network),
	_optimizer(p_optimizer),
	_gamma(p_gamma)
{
	_delta = tensor({ 1,1 });
}

Qlearning::~Qlearning()
= default;

/**
 * \brief Function returning discrete action for critic-only scenario
 * \param p_state state tensor in timestep t
 * \return tensor where Q-value with maximal value is set to 1. and the others to 0. (one-hot encoding)
 */
tensor& Qlearning::get_action(tensor* p_state)
{
	tensor& q_values = _network->forward(p_state);
	_action = tensor::zero_like(q_values);
	_action[q_values.max_index()[0]] = 1.f;

	return _action;
}

/**
 * \brief Learning rule Q(s0,a0) = Q(s0,a0) + alpha * (reward + gamma * Q_max[a](s1,a) - Q(s0,a0))
 * \param p_state state in timestep t
 * \param p_action action in timestep t
 * \param p_next_state state in timestep t+1
 * \param p_reward reward value
 * \param p_final flag indicating the last step of episode
 */
void Qlearning::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_network->backward(loss_function(p_state, p_action, p_next_state, p_reward, p_final));
	_optimizer->update();
}

/**
 * \brief Returns temporal difference error delta = reward + gamma * Q_max[a](s1,a) - Q(s0,a0)
 * \return TD error tensor with shape (1,1)
 */
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
