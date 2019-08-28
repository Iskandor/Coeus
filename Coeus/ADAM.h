#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) ADAM : public GradientAlgorithm
	{
	public:
		explicit ADAM(NeuralNetwork* p_network);
		~ADAM();

		void init(ICostFunction* p_cost_function, float p_alpha, float p_beta1 = 0.9, float p_beta2 = 0.999, float p_epsilon = 1e-8);
	};
}

