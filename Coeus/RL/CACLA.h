#pragma once
#include "NeuralNetwork.h"
#include "NetworkGradient.h"
#include "IUpdateRule.h"
#include "QuadraticCost.h"
#include "TD.h"

namespace Coeus
{
	class __declspec(dllexport) CACLA
	{
	public:
		CACLA(NeuralNetwork* p_critic, GRADIENT_RULE p_critic_rule, float p_critic_alpha, float p_gamma, 
			  NeuralNetwork* p_actor, GRADIENT_RULE p_actor_rule, float p_actor_alpha, float p_beta = 0.f);
		virtual ~CACLA();

		virtual void train(Tensor* p_state0, Tensor* p_action0, Tensor* p_state1, float p_reward, bool p_final);
		Tensor get_action(Tensor* p_state, float p_sigma = 1.0f) const;

	protected:
		TD*	_critic;
		QuadraticCost _mse;
		NeuralNetwork* _actor;
		NetworkGradient* _actor_gradient;
		IUpdateRule* _update_rule;

		float _actor_alpha;
		float _beta;
		float _var;
	};
}
