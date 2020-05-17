#pragma once
#include "neural_network.h"
#include "optimizer.h"

class COEUS_DLL_API CACLA
{
public:
	CACLA(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma);
	~CACLA();

	tensor& get_action(tensor* p_state) const;
	void train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final);

private:
	tensor& actor_loss_function(tensor* p_state, tensor* p_action);
	tensor& critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final);

	neural_network* _actor;
	optimizer* _actor_optimizer;
	neural_network* _critic;	
	optimizer* _critic_optimizer;
	float _gamma;
	float _delta;

	tensor _critic_loss;
	tensor _actor_loss;
};

