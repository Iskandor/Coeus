#include "CACLA.h"


CACLA::CACLA(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, const float p_gamma) :
	_actor(p_actor),
	_actor_optimizer(p_actor_optimizer)
{
	_critic = new TD(p_critic, p_critic_optimizer, p_gamma);
}

CACLA::~CACLA()
{
	delete _critic;
}

tensor& CACLA::get_action(tensor* p_state) const
{
	return _actor->forward(p_state);
}

void CACLA::train(tensor* p_state, tensor* p_action, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_critic->train(p_state, p_next_state, p_reward, p_final);
	if (_critic->delta()[0] > 0.f)
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