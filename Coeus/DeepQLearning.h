#pragma once
#include "QLearning.h"
#include "ReplayBuffer.h"
#include "GradientAlgorithm.h"
#include "BufferItems.h"

namespace Coeus
{
	class __declspec(dllexport) DeepQLearning
	{
	public:
		DeepQLearning(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm, float p_gamma, int p_size, int p_sample);
		~DeepQLearning();

		float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward, bool p_final) const;

	private:
		float calc_max_qa(Tensor* p_state) const;

		NeuralNetwork* _network;
		GradientAlgorithm* _gradient_algorithm;
		float _gamma;

		Tensor* _input;
		Tensor* _target;

		ReplayBuffer<DQItem>*	_replay_buffer;
		int _sample_size;
	};
}


