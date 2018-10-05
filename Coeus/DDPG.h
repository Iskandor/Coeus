#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"
#include "ReplayBuffer.h"

namespace Coeus
{
	// deep deterministic policy gradient
class __declspec(dllexport) DDPG
{
public:
	DDPG(NeuralNetwork* p_network_critic, BaseGradientAlgorithm* p_gradient_algorithm_critic, const double p_gamma,
		 NeuralNetwork* p_network_actor, BaseGradientAlgorithm* p_gradient_algorithm_actor, int p_buffer_size, int p_sample_size);
	virtual ~DDPG();

	double train(Tensor* p_state0, int p_action0, Tensor* p_state1, double p_reward);

private:
	double calc_max_qa(Tensor* p_state) const;

	NeuralNetwork* _network_actor;
	NeuralNetwork* _network_critic;
	NeuralNetwork* _network_actor_target;
	NeuralNetwork* _network_critic_target;

	BaseGradientAlgorithm* _gradient_algorithm_actor;
	BaseGradientAlgorithm* _gradient_algorithm_critic;
	double _gamma;

	ReplayBuffer* _buffer;
	int _sample_size;

	vector<Tensor*> _input;
	vector<Tensor*> _target;
};
}


