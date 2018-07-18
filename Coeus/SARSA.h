#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) SARSA
	{
	public:
		SARSA(NeuralNetwork* p_network, BaseGradientAlgorithm* p_gradient_algorithm, double p_gamma);
		~SARSA();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, int p_action1, double p_reward);

	private:
		NeuralNetwork* _network;
		BaseGradientAlgorithm* _gradient_algorithm;
		double _gamma;

		Tensor _target;
	};
}

