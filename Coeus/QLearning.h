#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) QLearning
	{
	public:
		QLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda = 0);
		virtual ~QLearning();

		float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward, bool p_final) const;

	private:
		float calc_max_qa(Tensor* p_state) const;

		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		IUpdateRule* _update_rule;

		float _alpha;
		float _gamma;
	};
}
