#pragma once
#include "NeuralNetwork.h"
#include "GradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) DoubleQLearning
	{
	public:
		DoubleQLearning(NeuralNetwork* p_network_a, GradientAlgorithm* p_gradient_algorithm_a, NeuralNetwork* p_network_b, GradientAlgorithm* p_gradient_algorithm_b, const float p_gamma);
		~DoubleQLearning();

		float train(Tensor* p_state0, int p_action0, Tensor* p_state1, float p_reward);

	protected:
		float calc_max_qa(Tensor* p_state, NeuralNetwork* p_network) const;

		NeuralNetwork* _network_a;
		GradientAlgorithm* _gradient_algorithm_a;
		NeuralNetwork* _network_b;
		GradientAlgorithm* _gradient_algorithm_b;
		float _gamma;

		Tensor _target;
	};
}


