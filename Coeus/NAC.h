/*
 * Natural Policy Gradient
 * 
 */
#pragma once
#include "GAE.h"

namespace Coeus
{
	class __declspec(dllexport) NAC
	{
	public:
		NAC(NeuralNetwork& p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, float p_lambda, NeuralNetwork& p_network_actor, float p_alpha_actor, float p_delta);
		~NAC();

		void add_sample(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, float p_r, const bool p_final);
		void train();

	private:
		NeuralNetwork		_network_actor;
		NetworkGradient*	_gradient_actor;
		IUpdateRule*		_rule_actor;
		Tensor				_actor_loss;
		map<string, Tensor> _gradient_estimate;
		map<string, Tensor> _update;
		
		GAE* _critic;

		vector<DQItem> _sample_buffer;
		float _delta;
		float _epsilon;
	};
}

