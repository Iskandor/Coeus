#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) ForwardModel
	{
	public:
		ForwardModel(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha);
		~ForwardModel();

		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1);
		void activate(Tensor* p_state, Tensor* p_action);

	private:
		NeuralNetwork*		_network;
		NetworkGradient*	_network_gradient;
		IUpdateRule*		_update_rule;

		Tensor*	_input;
	};
}

