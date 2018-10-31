#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) AdaMax : public BaseGradientAlgorithm
	{
	public:
		explicit AdaMax(NeuralNetwork* p_network);
		~AdaMax();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_beta1 = 0.9, double p_beta2 = 0.999, double p_epsilon = 1e-8);
	};
}

