#include "policy_gradient.h"

/**
 * \brief Policy gradient algorithm
 * \param p_network neural network approximating policy
 * \param p_optimizer network optimization algorithm
 */
policy_gradient::policy_gradient(neural_network* p_network, optimizer* p_optimizer) :
	_network(p_network),
	_optimizer(p_optimizer)
{
}

policy_gradient::~policy_gradient()
{
}

/**
 * \brief Learning rule pi(s,a) = pi(s,a) + alpha * delta * ln pi(s,a)
 * \param p_state state in timestep t
 * \param p_action action in timestep t 
 * \param p_delta temporal difference error
 */
void policy_gradient::train(tensor* p_state, tensor* p_action, tensor& p_delta)
{
	_network->backward(loss_function(p_state, p_action, p_delta));
	_optimizer->update();
}

/**
 * \brief Returns probabilities of each action according to pi(s,a)
 * \param p_state state in timestep t
 * \return tensor of probabilities pi(s,a)
 */
tensor& policy_gradient::get_action(tensor* p_state) const
{
	return _network->forward(p_state);
}

tensor& policy_gradient::loss_function(tensor* p_state, tensor* p_action, tensor& p_delta)
{
	tensor& action = _network->forward(p_state);
	const int action_index = p_action->max_index()[0];

	_loss.resize({ 1, action.size() });
	_loss[action_index] = p_delta[0] / action[action_index];

	return _loss;
}