/*
Generalized Advantage Estimator
https://arxiv.org/pdf/1506.02438.pdf

*/

#pragma once
#include <vector>
#include "TD.h"
#include "BufferItems.h"

namespace Coeus {
	class __declspec(dllexport) GAE
	{
	public:
		GAE(NeuralNetwork* p_network, float p_gamma, float p_lambda);
		GAE(NeuralNetwork* p_network, GRADIENT_RULE p_rule, float p_alpha, float p_gamma, float p_lambda);
		~GAE();

		Tensor&		get_advantages(vector<DQItem> &p_sample);
		void		train(vector<DQItem> &p_sample);

		Gradient&	get_gradient(Tensor* p_state0, float p_return) const;


	private:
		void activate(vector<DQItem> &p_sample) const;
		
		Tensor				_input_buffer_s0;
		Tensor				_returns;
		Tensor				_advantages;
		NeuralNetwork*		_network;
		NetworkGradient*	_network_gradient;
		IUpdateRule*		_update_rule;
		
		float	_gamma;
		float	_lambda;

	};
}


