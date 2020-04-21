#pragma once
#include "tensor.h"
#include "neural_network.h"
#include "optimizer.h"

class __declspec(dllexport) SARSA
{
public:
	SARSA(neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma);
	~SARSA();

	tensor& get_action(tensor* p_state);
	void train(tensor* p_state, tensor* p_action, tensor* p_next_state, tensor* p_next_action, float p_reward, bool p_final);

private:
	tensor&	critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, tensor* p_next_action, float p_reward, bool p_final);

	neural_network* _critic;
	optimizer*		_critic_optimizer;
	float			_gamma;

	tensor			_action;
	tensor			_critic_loss;
};

