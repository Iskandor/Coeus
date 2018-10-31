#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) AMSGrad : public BaseGradientAlgorithm
	{
	public:
		explicit AMSGrad(NeuralNetwork* p_network);
		~AMSGrad();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_beta1 = 0.9, double p_beta2 = 0.999, double p_epsilon = 1e-8);
	};
}


