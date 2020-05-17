#pragma once
#include "Qlearning.h"
#include "replay_buffer.h"

class COEUS_DLL_API DQN : public Qlearning
{
public:
	DQN(neural_network* p_critic, optimizer* p_critic_optimizer, float p_gamma, int p_memory_size, int p_sample, int p_target_update_frequency);
	~DQN();

	void train(tensor* p_state, tensor* p_action, tensor* p_next_state, float p_reward, bool p_final) override;

private:
	tensor& critic_loss_function(tensor* p_state, tensor* p_action, tensor* p_next_state, tensor* p_reward, tensor* p_mask);
	void	process_sample();

	tensor _batch_state;
	tensor _batch_action;
	tensor _batch_next_state;
	tensor _batch_reward;
	tensor _batch_mask;

	replay_buffer<mdp_transition> *_memory;
	int _sample_size;

	neural_network _critic_target;
	int _target_update_frequency;
	int _target_update_step;
};

