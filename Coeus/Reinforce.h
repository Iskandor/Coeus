#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"
#include "ReinforceRule.h"
#include "ADAMRule.h"

namespace Coeus
{
	class __declspec(dllexport) Reinforce
	{
	public:
		Reinforce(NeuralNetwork* p_network, float p_alpha);
		~Reinforce();

		void train(Tensor* p_state, int p_action, float p_delta);

	private:
		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		//ReinforceRule* _update_rule;
		ADAMRule* _update_rule;

		float _alpha;
		int _t;
	};
}

