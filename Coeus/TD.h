#pragma once
#include "NeuralNetwork.h"
#include "GeneralTDRule.h"

namespace Coeus
{

	class __declspec(dllexport) TD
	{
	public:
		TD(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, double p_alpha, double p_gamma, double p_lambda = 0);
		~TD();

		double train(Tensor* p_state0, Tensor* p_state1, double p_reward) const;

	private:
		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		GeneralTDRule* _update_rule;
		double _gamma;
	};

}
