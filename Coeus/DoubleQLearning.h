#pragma once
#include "NeuralNetwork.h"
#include "BaseGradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) DoubleQLearning
	{
	public:
		DoubleQLearning(NeuralNetwork* p_network_a, BaseGradientAlgorithm* p_gradient_algorithm_a, NeuralNetwork* p_network_b, BaseGradientAlgorithm* p_gradient_algorithm_b, const double p_gamma);
		~DoubleQLearning();

		double train(Tensor* p_state0, int p_action0, Tensor* p_state1, double p_reward);

	protected:
		double calc_max_qa(Tensor* p_state, NeuralNetwork* p_network) const;

		NeuralNetwork* _network_a;
		BaseGradientAlgorithm* _gradient_algorithm_a;
		NeuralNetwork* _network_b;
		BaseGradientAlgorithm* _gradient_algorithm_b;
		double _gamma;

		Tensor _target;
	};
}


