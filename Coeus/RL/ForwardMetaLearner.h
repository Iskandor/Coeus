#pragma once
#include "ForwardModel.h"

namespace Coeus
{
	class __declspec(dllexport) ForwardMetaLearner
	{
	public:
		ForwardMetaLearner(ForwardModel* p_forward_model, NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha);
		~ForwardMetaLearner();

		float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1);
		void activate(Tensor* p_state, Tensor* p_action);

	private:
		ForwardModel*		_forward_model;
		NeuralNetwork*		_network;
		NetworkGradient*	_network_gradient;
		IUpdateRule*		_update_rule;

		Tensor*	_input;
	};
}
