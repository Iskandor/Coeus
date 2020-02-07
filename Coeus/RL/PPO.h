#pragma once
#include "GAE.h"
#include "PolicyGradient.h"

namespace Coeus
{
	class __declspec(dllexport) PPO
	{
	public:
		PPO(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, float p_lambda, NeuralNetwork* p_network_actor, GRADIENT_RULE p_rule_actor, float p_alpha_actor, size_t p_trajectory_size);
		~PPO();

		void train(Tensor* p_state0, Tensor* p_action, Tensor* p_state1, float p_reward, bool p_final);
		Tensor get_action(Tensor* p_state) const;
		
	private:
		float clip(float p_value, float p_lower_bound, float p_upper_bound) const;
		float _epsilon;
		
		size_t			_trajectory_size;
		vector<DQItem>	_trajectory;

		GAE*				_critic;
		NeuralNetwork*		_actor_old;
		NeuralNetwork*		_actor_new;
		NetworkGradient*	_actor_gradient;
		IUpdateRule*		_actor_old_rule;
		IUpdateRule*		_actor_new_rule;

		PolicyGradient*		_policy_gradient;
		
		
	};
}

