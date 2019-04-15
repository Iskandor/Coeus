#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "GradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) BackProp : public GradientAlgorithm
	{
	public:
		explicit BackProp(NeuralNetwork* p_network);
		~BackProp();

		void init(ICostFunction* p_cost_function, float p_alpha, float p_momentum = 0, bool p_nesterov = false);
	};
}
