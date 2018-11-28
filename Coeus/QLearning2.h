#pragma once
#include "NeuralNetwork.h"
#include "QLearningRule.h"

namespace Coeus
{
	class __declspec(dllexport) QLearning2
	{
	public:
		QLearning2(NeuralNetwork* p_network, double p_alpha, double p_gamma, double p_lambda = 0);
		virtual ~QLearning2();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, double p_reward) const;
		void reset_traces() const;

	private:
		double calc_max_qa(Tensor* p_state) const;

		NeuralNetwork* _network;
		NetworkGradient* _network_gradient;
		QLearningRule* _update_rule;
		double _gamma;
	};
}
