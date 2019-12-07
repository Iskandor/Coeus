#pragma once
#include "NeuralNetwork.h"

namespace Coeus {

	class __declspec(dllexport) IActorCritic
	{
	public:
		IActorCritic(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha);
		~IActorCritic();

	private:

	};

}


