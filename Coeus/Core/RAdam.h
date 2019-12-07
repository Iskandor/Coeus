#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) RADAM : public GradientAlgorithm
	{
	public:
		explicit RADAM(NeuralNetwork* p_network);
		~RADAM();

		void init(ICostFunction* p_cost_function, float p_alpha, float p_beta1 = 0.9, float p_beta2 = 0.999);
	};
}

