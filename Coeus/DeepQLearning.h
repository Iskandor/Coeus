#pragma once
#include "QLearning.h"
#include "ReplayBuffer.h"

namespace Coeus
{
	class __declspec(dllexport) DeepQLearning
	{
	public:
		DeepQLearning(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, double p_gamma, int p_size, int p_sample);
		~DeepQLearning();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, double p_reward, bool p_final);

	private:
		double calc_max_qa(Tensor* p_state);

		NeuralNetwork* _network;
		BaseGradientAlgorithm* _gradient_algorithm;
		double _gamma;

		vector<Tensor*> *_input;
		vector<Tensor*> *_target;

		ReplayBuffer*	_replay_buffer;
		int _sample_size;
	};
}


