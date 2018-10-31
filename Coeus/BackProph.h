#pragma once
#include "BaseGradientAlgorithm.h"


namespace Coeus {

	class __declspec(dllexport) BackProp : public BaseGradientAlgorithm
	{
	public:
		explicit BackProp(NeuralNetwork* p_network);
		~BackProp();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_momentum = 0, bool p_nesterov = false);
	};
}
