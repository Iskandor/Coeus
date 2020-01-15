#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "ReplayBuffer.h"
#include "BufferItems.h"

namespace Coeus
{
// deep deterministic policy gradient
class __declspec(dllexport) DDPG
{
public:
	DDPG(NeuralNetwork* p_network_critic, GRADIENT_RULE p_critic_rule, float p_critic_alpha, float p_gamma,
		 NeuralNetwork* p_network_actor, GRADIENT_RULE p_actor_rule, float p_actor_alpha, int p_buffer_size, int p_sample_size);
	virtual ~DDPG();

	void train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, float p_reward, bool p_final) const;
	Tensor get_action(Tensor* p_state, float p_sigma) const;

private:	
	float calc_max_qa(Tensor* p_state) const;

	NeuralNetwork* _network_actor;
	NetworkGradient* _network_actor_gradient;
	IUpdateRule* _update_rule_actor;
	NeuralNetwork* _network_actor_target;

	NeuralNetwork* _network_critic;
	NetworkGradient* _network_critic_gradient;
	IUpdateRule* _update_rule_critic;
	NeuralNetwork* _network_critic_target;

	float _gamma;

	ReplayBuffer<DQItem>* _buffer;
	int _sample_size;

	Tensor* _critic_input;
	Tensor* _critic_input2;
	Tensor* _actor_input;
	Tensor* _target;
};
}


