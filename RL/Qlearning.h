#pragma once
#include "neural_network.h"
#include "optimizer.h"

class COEUS_DLL_API Qlearning
{
public:
	Qlearning(neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma);
	virtual ~Qlearning();

	tensor& get_action(tensor* p_state);
	virtual void train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final);
	tensor&	delta();

protected:	

	neural_network* _network;
	optimizer*		_optimizer;
	float			_gamma;

	tensor			_action;
	tensor			_delta;
	tensor			_loss;

private:
	tensor&	loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final);
};

