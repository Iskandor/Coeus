#pragma once
#include "NeuralNetwork.h"
#include "GeneralTDRule.h"
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) QLearning
	{
	public:
		QLearning(NeuralNetwork* p_network, GradientAlgorithm* p_optimizer, float p_gamma, float p_alpha = 1);
		QLearning(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda = 0);
		virtual ~QLearning();

		float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward) const;
		void reset_traces() const;

	private:
		float calc_max_qa(Tensor* p_state) const;

		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		GeneralTDRule* _update_rule;
		GradientAlgorithm* _optimizer;
		float _alpha;
		float _gamma;
	};
}
