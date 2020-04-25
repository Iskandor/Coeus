#pragma once
#include "neural_network.h"
#include "optimizer.h"
#include "replay_buffer.h"
#include "forward_model.h"

class __declspec(dllexport) DDPG
{
public:
	DDPG(neural_network* p_actor, optimizer* p_actor_optimizer, neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma, int p_memory_size, int p_sample, float p_tau = 1e-3f);
	~DDPG();

	tensor& get_action(tensor* p_state) const;
	void train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final);
	void add_motivation(forward_model* p_motivation);

private:
	void process_sample();
	tensor& actor_loss_function();
	tensor& critic_loss_function();

	neural_network* _actor;
	neural_network _actor_target;
	neural_network* _critic;
	neural_network _critic_target;

	float _gamma;
	float _tau;

	optimizer* _actor_optimizer;
	optimizer* _critic_optimizer;

	replay_buffer<mdp_transition>* _memory;
	int _sample_size;

	tensor batch_state;
	tensor batch_action;
	tensor batch_next_state;
	tensor batch_reward;
	tensor batch_mask;

	tensor _critic_loss;
	tensor _actor_loss;

	forward_model* _motivation;
};

