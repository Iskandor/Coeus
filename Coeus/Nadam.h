#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) Nadam : public BaseGradientAlgorithm
	{
	public:
		explicit Nadam(NeuralNetwork* p_network);
		~Nadam();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_beta1 = 0.9, double p_beta2 = 0.999, double p_epsilon = 1e-8);
	};
}

