#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) PowerSign : public GradientAlgorithm
	{
	public:
		PowerSign(NeuralNetwork* p_network);
		~PowerSign();

		void init(ICostFunction* p_cost_function, float p_alpha = exp(1));
	};
}

