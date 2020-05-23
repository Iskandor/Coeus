#pragma once
#include "neural_network.h"
#include "optimizer.h"

class COEUS_DLL_API AC
{
public:
	AC(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma);
	~AC();

	tensor& get_action(tensor* p_state) const;
	void train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final);

private:
	tensor& actor_loss_function(tensor* p_state, tensor* p_action, tensor& p_delta);
	tensor& critic_loss_function(tensor* p_state, tensor* p_next_state, float p_reward, bool p_final);

	neural_network* _actor;
	optimizer* _actor_optimizer;
	neural_network* _critic;
	optimizer* _critic_optimizer;

	float _gamma;

	tensor _critic_loss;
	tensor _actor_loss;
};

