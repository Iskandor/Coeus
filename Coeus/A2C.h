#pragma once
#include "IEnvironment.h"
#include "NeuralNetwork.h"
#include "GAE.h"
#include "PolicyGradient.h"

namespace Coeus
{
	class __declspec(dllexport) A2C
	{
	public:
		A2C(vector<IEnvironment*> &p_env_array,
			NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma, float p_lambda,
			NeuralNetwork* p_actor, GRADIENT_RULE p_actor_update_rule, float p_actor_alpha);
		~A2C();
		
		void train(int p_rollout_size) const;

	private:
		
		vector<IEnvironment*> _env_array;

		NeuralNetwork*		_actor;
		NeuralNetwork**		_actor_array;
		map<string, Tensor>	_actor_d_gradient;
		map<string, Tensor>* _actor_d_gradient_array;
		PolicyGradient**	_policy_gradient;
		IUpdateRule*		_actor_rule;
		
		NeuralNetwork*		_critic;
		NeuralNetwork**		_critic_array;
		map<string, Tensor>	_critic_d_gradient;
		map<string, Tensor>* _critic_d_gradient_array;
		GAE**				_advantage_estimation;
		IUpdateRule*		_critic_rule;
		

		vector<DQItem>* _sample_buffer;
	};
}

