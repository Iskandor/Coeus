#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) PowerSign : public BaseGradientAlgorithm
	{
	public:
		PowerSign(NeuralNetwork* p_network);
		~PowerSign();

		void init(ICostFunction* p_cost_function, double p_alpha = exp(1));
	};
}

