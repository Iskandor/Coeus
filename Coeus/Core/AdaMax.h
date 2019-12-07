#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) AdaMax : public GradientAlgorithm
	{
	public:
		explicit AdaMax(NeuralNetwork* p_network);
		~AdaMax();

		void init(ICostFunction* p_cost_function, float p_alpha, float p_beta1 = 0.9, float p_beta2 = 0.999, float p_epsilon = 1e-8);
	};
}
