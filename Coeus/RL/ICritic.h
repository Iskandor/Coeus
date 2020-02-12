#pragma once
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "RuleFactory.h"

namespace Coeus {

	class __declspec(dllexport) ICritic
	{
	public:
		ICritic(NeuralNetwork* p_network, const GRADIENT_RULE p_rule, const float p_alpha)
		{
			_network = p_network;
			_network_gradient = new NetworkGradient(p_network);
			_update_rule = RuleFactory::create_rule(p_rule, p_network, p_alpha);
		}

		virtual ~ICritic()
		{
			delete _network_gradient;
			delete _update_rule;
		}

		virtual float train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final) = 0;

	protected:
		NeuralNetwork*		_network;
		NetworkGradient*	_network_gradient;
		IUpdateRule*		_update_rule;
	};	
}


