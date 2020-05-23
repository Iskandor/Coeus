#include "AC.h"

AC::AC(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma) :
	_actor(p_actor),
	_actor_optimizer(p_actor_optimizer),
	_critic(p_critic),
	_critic_optimizer(p_critic_optimizer),
	_gamma(p_gamma)
{
	_critic_loss.resize({ 1,1 });
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
	tensor& delta = critic_loss_function(p_state, p_next_state, p_reward, p_final);
	_critic->backward(delta);
	_critic_optimizer->update();
	_actor->backward(actor_loss_function(p_state, p_action, delta));
	_actor_optimizer->update();
}

tensor& AC::actor_loss_function(tensor* p_state, tensor* p_action, tensor& p_delta)
{
	tensor& action = _actor->forward(p_state);
	const int action_index = p_action->max_index()[0];
	
	_actor_loss.resize({1, action.size()});
	_actor_loss[action_index] = p_delta[0] / action[action_index];	
	
	return _actor_loss;
}

tensor& AC::critic_loss_function(tensor* p_state, tensor* p_next_state, float p_reward, bool p_final)
{
	const float V1 = _critic->forward(p_next_state)[0];
	const float V0 = _critic->forward(p_state)[0];

	float delta = V0;

	if (p_final)
	{
		delta -= p_reward;
	}
	else
	{
		delta -= p_reward + _gamma * V1;
	}
	_critic_loss[0] = delta;
	
	return _critic_loss;
}
