/*
 * Natural Policy Gradient
 * 
 */
#pragma once
#include "GAE.h"
#include "PolicyGradient.h"

namespace Coeus
{
	class __declspec(dllexport) NAC
	{
	public:
		NAC(NeuralNetwork* p_network_critic, GRADIENT_RULE p_rule_critic, float p_alpha_critic, float p_gamma, float p_lambda, NeuralNetwork* p_network_actor, float p_alpha_actor, float p_delta);
		~NAC();

		void add_sample(Tensor* p_s0, Tensor* p_a, Tensor* p_s1, float p_r, bool p_final);
		void train();

	private:
		NeuralNetwork*		_network_actor;
		NeuralNetwork*		_network_critic;
		PolicyGradient*		_gradient_actor;
		IUpdateRule*		_rule_actor;
		IUpdateRule*		_rule_critic;
		map<string, Tensor> _fim;
		map<string, Tensor> _inv_fim;
		map<string, Tensor> _update;
		
		GAE* _critic;

		vector<DQItem> _sample_buffer;
		float _delta;
		float _epsilon;
	};
}

