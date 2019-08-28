#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "ADAMRule.h"

namespace Coeus
{
	class __declspec(dllexport) Actor
	{
	public:
		Actor(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha);
		~Actor();

		void train(Tensor* p_state, int p_action, float p_delta) const;

	private:
		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		IUpdateRule* _update_rule;

		float _alpha;
	};
}

