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
		GAE(NeuralNetwork* p_network, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda);
		~GAE();

		void set_sample(vector<DQItem> &p_sample);
		float get_advantage();
		void train();


	private:
		NeuralNetwork *_network;
		TD *_value_estimator;

		vector<DQItem> _sample_buffer;

		float _gamma;
		float _lambda;

	};
}


