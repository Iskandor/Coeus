#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) DoubleQLearning
	{
	public:
		DoubleQLearning(NeuralNetwork* p_network_a, NeuralNetwork* p_network_b, GRADIENT_RULE p_grad_rule, float p_alpha, float p_gamma, float p_lambda = 0);
		~DoubleQLearning();

		float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward, bool p_finished) const;
		Tensor* get_output(Tensor* p_state) const;

	protected:
		int calc_max_qa_index(Tensor* p_state, NeuralNetwork* p_network) const;

		NeuralNetwork* _network_a;
		NeuralNetwork* _network_b;
		NetworkGradient* _network_gradient_a;
		NetworkGradient* _network_gradient_b;
		IUpdateRule* _update_rule_a;
		IUpdateRule* _update_rule_b;

		float _alpha;
		float _gamma;

		int _action_count;
		Tensor* _output;
	};
}


