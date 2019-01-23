#pragma once
#include "QLearning.h"
#include "ReplayBuffer.h"
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) DeepQLearning
	{
	public:
		DeepQLearning(NeuralNetwork* p_network, GradientAlgorithm* p_gradient_algorithm, double p_gamma, int p_size, int p_sample);
		~DeepQLearning();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, double p_reward, bool p_final) const;

	private:
		double calc_max_qa(Tensor* p_state) const;

		NeuralNetwork* _network;
		GradientAlgorithm* _gradient_algorithm;
		double _gamma;

		vector<Tensor*> *_input;
		vector<Tensor*> *_target;

		ReplayBuffer*	_replay_buffer;
		int _sample_size;
	};
}


