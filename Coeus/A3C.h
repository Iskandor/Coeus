#pragma once
#include <vector>
#include "IEnvironment.h"
#include "NeuralNetwork.h"
#include "PolicyGradient.h"
#include "GAE.h"

namespace Coeus
{
	class __declspec(dllexport) A3C
	{
	public:
		A3C(std::vector<IEnvironment*> &p_env_array, int p_t_max,
			NeuralNetwork* p_critic, GRADIENT_RULE p_critic_update_rule, float p_critic_alpha, float p_gamma, float p_lambda,
			NeuralNetwork* p_actor, GRADIENT_RULE p_actor_update_rule, float p_actor_alpha);
		~A3C();
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
		GAE**				_advantage_estimation;
		IUpdateRule*		_critic_rule;

		vector<DQItem>* _sample_buffer;

		int		_t_max;
		int*	_t;
	};
}

