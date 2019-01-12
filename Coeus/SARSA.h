#pragma once
#include "NeuralNetwork.h"
#include "GeneralTDRule.h"

namespace Coeus {

	class __declspec(dllexport) SARSA
	{
	public:
		SARSA(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, double p_alpha, double p_gamma, double p_lambda = 0);
		~SARSA();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, int p_action1, double p_reward) const;
		void reset_traces() const;

	private:
		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		GeneralTDRule* _update_rule;
		double _gamma;
	};
}

