#pragma once
#include "NeuralNetwork.h"

namespace Coeus {

	class __declspec(dllexport) IActorCritic
	{
	public:
		IActorCritic(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha);
		~IActorCritic();

		virtual void train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final) = 0;
		virtual Tensor get_action(Tensor* p_state) = 0;

	private:

	};

}


