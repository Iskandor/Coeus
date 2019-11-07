#pragma once
#include "IEnvironment.h"
#include "NeuralNetwork.h"
#include "GAE.h"
#include "Actor.h"

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
		ParamModel**		_actor_param_array;
		Actor**				_policy_gradient;
		
		NeuralNetwork*	_critic;
		NeuralNetwork**	_critic_array;
		ParamModel**	_critic_param_array;
		GAE**			_advantage_estimation;

		vector<DQItem>* _sample_buffer;
	};
}

