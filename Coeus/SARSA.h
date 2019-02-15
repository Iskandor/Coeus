#pragma once
#include "NeuralNetwork.h"
#include "GeneralTDRule.h"
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) SARSA
	{
	public:
		SARSA(NeuralNetwork* p_network, GradientAlgorithm* p_optimizer, double p_gamma, double p_alpha = 1);
		SARSA(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, double p_alpha, double p_gamma, double p_lambda = 0);
		~SARSA();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, int p_action1, double p_reward, bool p_final) const;
		void reset_traces() const;

	private:
		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		GeneralTDRule* _update_rule;
		GradientAlgorithm* _optimizer;
		double _alpha;
		double _gamma;
	};
}

