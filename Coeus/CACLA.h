#pragma once
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include "QuadraticCost.h"

namespace Coeus
{
	class __declspec(dllexport) CACLA
	{
	public:
		CACLA(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha, float p_beta = 0.f);
		~CACLA();

		void train(Tensor* p_state, Tensor* p_action, float p_delta);
		Tensor get_action(Tensor* p_state, float p_sigma = 1.0f) const;

	private:
		QuadraticCost _mse;
		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		IUpdateRule* _update_rule;

		float _alpha;
		float _beta;
		float _var;
	};
}
