#include "QAC.h"

QAC::QAC(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma)
{
	_actor = new policy_gradient(p_actor, p_actor_optimizer);
	_critic = new Qlearning(p_critic, p_critic_optimizer, p_gamma);
}

QAC::~QAC()
{
	delete _actor;
	delete _critic;
}

tensor& QAC::get_action(tensor* p_state) const
{
	return _actor->get_action(p_state);
}

void QAC::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final) const
{
	_critic->train(p_state, p_action, p_next_state, p_reward, p_final);
	_actor->train(p_state, p_action, _critic->delta());
}
