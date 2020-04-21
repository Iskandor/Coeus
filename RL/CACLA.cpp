#include "CACLA.h"


CACLA::CACLA(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma) :
	_actor(p_actor),
	_actor_optimizer(p_actor_optimizer),
	_critic(p_critic),
	_critic_optimizer(p_critic_optimizer),
	_gamma(p_gamma)
{
	_critic_loss = tensor({ 1,1 });
}

CACLA::~CACLA()
= default;

tensor& CACLA::get_action(tensor* p_state) const
{
	return _actor->forward(p_state);
}

void CACLA::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_critic->backward(critic_loss_function(p_state, p_action, p_next_state, p_reward, p_final));
	_critic_optimizer->update();
	if (_delta > 0)
	{
		_actor->backward(actor_loss_function(p_state, p_action));
		_actor_optimizer->update();
	}
}

tensor& CACLA::actor_loss_function(tensor* p_state, tensor* p_action)
{
	_actor_loss = _actor->forward(p_state) - *p_action;
	return _actor_loss;
}

tensor& CACLA::critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	const float Vs1 = _critic->forward(p_next_state)[0];
	const float Vs0 = _critic->forward(p_state)[0];

	if (p_final)
	{
		_delta = p_reward - Vs0;
	}
	else
	{
		_delta = p_reward + _gamma * Vs1 - Vs0;
	}
	_critic_loss[0] = -_delta;

	return _critic_loss;
}