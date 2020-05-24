#pragma once
#include "optimizer.h"

class COEUS_DLL_API policy_gradient
{
public:
	policy_gradient(neural_network* p_network, optimizer* p_optimizer);
	~policy_gradient();
	
	void	train(tensor* p_state, tensor* p_action, tensor& p_delta);
	tensor& get_action(tensor* p_state) const;

private:
	tensor& loss_function(tensor* p_state, tensor* p_action, tensor& p_delta);

	neural_network* _network;
	optimizer*		_optimizer;

	tensor	_loss;
};
