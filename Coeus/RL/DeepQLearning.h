#pragma once
#include "QLearning.h"
#include "ReplayBuffer.h"
#include "GradientAlgorithm.h"
#include "BufferItems.h"

namespace Coeus
{
	class __declspec(dllexport) DeepQLearning : QLearning
	{
	public:
		DeepQLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, int p_size, int p_sample, int p_target_network_update);
		~DeepQLearning();

		float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward, bool p_final) override;

	protected:
		Tensor	_input_s0;
		Tensor	_input_s1;
		Tensor	_target;
		Tensor	_max_qa;

		ReplayBuffer<DQItem>*	_replay_buffer;
		int _sample_size;
		int _target_network_update;
		int _target_network_update_t;

		NeuralNetwork* _target_network;

	private:
		void calc_max_qa(int p_sample);
	};
}


