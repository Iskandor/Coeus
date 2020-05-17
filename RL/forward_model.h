#pragma once
#include "neural_network.h"
#include "optimizer.h"
#include "loss_functions.h"

class __declspec(dllexport) forward_model
{
public:
	forward_model(neural_network* p_network, optimizer* p_optimizer);
	~forward_model();

	void	train(tensor* p_state, tensor* p_action, tensor* p_next_state);
	tensor&	reward(tensor* p_state, tensor* p_action, tensor* p_next_state);
	tensor&	error(tensor* p_state, tensor* p_action, tensor* p_next_state);

private:
	neural_network* _network;
	optimizer*		_optimizer;

	tensor _input;
	tensor _error;
	tensor _reward;
	mse_function _loss_function;
};

