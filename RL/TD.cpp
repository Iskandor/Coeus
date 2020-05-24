#include "TD.h"

TD::TD(neural_network* p_network, optimizer* p_optimizer, const float p_gamma) :
	_network(p_network),
	_optimizer(p_optimizer),
	_gamma(p_gamma)
{
	_loss = tensor({ 1,1 });
}

TD::~TD()
{
}

void TD::train(tensor* p_state, tensor* p_next_state, const float p_reward, const bool p_final)
{
	_network->backward(loss_function(p_state, p_next_state, p_reward, p_final));
	_optimizer->update();
}

tensor& TD::delta()
{
	return _loss;
}

tensor& TD::loss_function(tensor* p_state, tensor* p_next_state, const float p_reward, const bool p_final)
{
	const float V1 = _network->forward(p_next_state)[0];
	const float V0 = _network->forward(p_state)[0];

	float delta = V0;

	if (p_final)
	{
		delta -= p_reward;
	}
	else
	{
		delta -= p_reward + _gamma * V1;
	}
	_loss[0] = delta;	
	return _loss;
}
