#pragma once
#include "neural_network.h"
#include "optimizer.h"
#include "policy_gradient.h"
#include "TD.h"

class COEUS_DLL_API AC
{
public:
	AC(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma);
	~AC();

	tensor& get_action(tensor* p_state) const;
	void train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final) const;

private:
	policy_gradient*	_actor;
	TD*					_critic;

};

