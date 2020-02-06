#pragma once
#include "GAE.h"

namespace Coeus
{
	class __declspec(dllexport) PPO
	{
	public:
		PPO(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, float p_lambda, NeuralNetwork* p_network_actor, GRADIENT_RULE p_rule_actor, float p_alpha_actor, size_t p_trajectory_size);
		~PPO();

		void train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final);
		
	private:
		size_t			_trajectory_size;
		vector<DQItem>	_trajectory;

		GAE*				_critic;
		NeuralNetwork*		_actor;
		NetworkGradient*	_actor_gradient;
		IUpdateRule*		_actor_rule;
		
	};
}

