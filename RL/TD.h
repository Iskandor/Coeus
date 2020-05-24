#pragma once
#include "optimizer.h"

class COEUS_DLL_API TD
{
public:
	TD(neural_network* p_network, optimizer* p_optimizer, float p_gamma);
	~TD();

	void	train(tensor* p_state, tensor* p_next_state, float p_reward, bool p_final);
	tensor&	delta();
	
private:
	tensor& loss_function(tensor* p_state, tensor* p_next_state, float p_reward, bool p_final);
	
	neural_network* _network;
	optimizer*		_optimizer;
	float	_gamma;
	tensor	_loss;
};
