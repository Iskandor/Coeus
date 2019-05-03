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
	DDPG(NeuralNetwork* p_network_critic, GradientAlgorithm* p_gradient_algorithm_critic, const float p_gamma,
		 NeuralNetwork* p_network_actor, GradientAlgorithm* p_gradient_algorithm_actor, int p_buffer_size, int p_sample_size);
	virtual ~DDPG();

	float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward);

private:
	float calc_max_qa(Tensor* p_state) const;

	NeuralNetwork* _network_actor;
	NeuralNetwork* _network_critic;
	NeuralNetwork* _network_actor_target;
	NeuralNetwork* _network_critic_target;

	GradientAlgorithm* _gradient_algorithm_actor;
	GradientAlgorithm* _gradient_algorithm_critic;
	float _gamma;

	ReplayBuffer<DQItem>* _buffer;
	int _sample_size;

	vector<Tensor*> _input;
	vector<Tensor*> _target;
};
}


