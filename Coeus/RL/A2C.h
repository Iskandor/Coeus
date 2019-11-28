#pragma once
#include "IEnvironment.h"
#include "NeuralNetwork.h"
#include "GAE.h"
#include "PolicyGradient.h"
#include "Gradient.h"

namespace Coeus
{
	class __declspec(dllexport) A2C
	{
	public:
		A2C(vector<IEnvironment*> &p_env_array,
			NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma,
			NeuralNetwork* p_actor, GRADIENT_RULE p_actor_update_rule, float p_actor_alpha);
		~A2C();
		
		void train(int p_rollout_size);

	private:
		
		vector<IEnvironment*> _env_array;

		NeuralNetwork*		_actor;
		NeuralNetwork**		_actor_array;
		Gradient			_actor_d_gradient;
		Gradient*			_actor_d_gradient_array;
		PolicyGradient**	_policy_gradient;
		IUpdateRule*		_actor_rule;
		
		NeuralNetwork*		_critic;
		NeuralNetwork**		_critic_array;
		Gradient			_critic_d_gradient;
		Gradient*			_critic_d_gradient_array;
		NetworkGradient**	_critic_gradient_array;
		IUpdateRule*		_critic_rule;
		
		float _gamma;
		vector<DQItem>* _sample_buffer;
	};
}

