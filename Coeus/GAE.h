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
		~GAE();

		void set_sample(vector<DQItem> &p_sample);
		vector<float> get_advantages();
		
		map<string, Tensor>& get_gradient(Tensor* p_state0, float p_advantage) const;


	private:
		NeuralNetwork*		_network;
		NetworkGradient*	_network_gradient;

		vector<DQItem> _sample_buffer;

		float _gamma;
		float _lambda;

	};
}


