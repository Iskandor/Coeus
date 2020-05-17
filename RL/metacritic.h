#pragma once
#include "forward_model.h"
#include "loss_functions.h"
#include "neural_network.h"
#include "optimizer.h"

class COEUS_DLL_API metacritic
{
public:
	metacritic(neural_network* p_network, optimizer* p_optimizer, forward_model* p_forward_model, float p_sigma);
	~metacritic();

	void	train(tensor* p_state, tensor* p_action, tensor* p_next_state);
	tensor&	reward(tensor* p_state, tensor* p_action, tensor* p_next_state);
	tensor&	error(tensor* p_state, tensor* p_action);

private:
	neural_network* _network;
	optimizer*		_optimizer;
	forward_model*	_forward_model;

	tensor _input;
	tensor _error;
	tensor _reward;
	mse_function _loss_function;
	float _sigma;
};
