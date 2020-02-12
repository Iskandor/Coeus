#pragma once
#include "NeuralNetwork.h"
#include "TD.h"
#include "PolicyGradient.h"

namespace Coeus
{
	class __declspec(dllexport) ActorCritic
	{
	public:
		ActorCritic(ICritic* p_critic, NeuralNetwork* p_network_actor, GRADIENT_RULE p_rule_actor, float p_alpha_actor);
		~ActorCritic();

		void train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, float p_reward, bool p_final) const;

	private:
		ICritic*	_critic;
		NeuralNetwork* _network_actor;			
		PolicyGradient* _actor;

		IUpdateRule* _rule_actor;
		
	};
}


