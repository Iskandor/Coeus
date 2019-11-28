#pragma once
#include <vector>
#include "IEnvironment.h"
#include "NeuralNetwork.h"
#include "A2C.h"

namespace Coeus
{
	class __declspec(dllexport) A3C : public A2C
	{
	public:
		A3C(std::vector<IEnvironment*> &p_env_array,
			NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma,
			NeuralNetwork* p_actor, GRADIENT_RULE p_actor_update_rule, float p_actor_alpha);
		~A3C();

		void train(int p_t_max, int p_T_max = 1) override;
	};
}

