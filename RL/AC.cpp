#include "AC.h"

AC::AC(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma) :
	_actor(p_actor),
	_actor_optimizer(p_actor_optimizer),
	_critic(p_critic),
	_critic_optimizer(p_critic_optimizer),
	_gamma(p_gamma)
{

}

AC::~AC()
{
}

tensor& AC::get_action(tensor* p_state) const
{
	return _actor->forward(p_state);
}

void AC::train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final)
{

}

tensor& AC::actor_loss_function()
{
	return _actor_loss;
}

tensor& AC::critic_loss_function()
{
	return _critic_loss;
}
