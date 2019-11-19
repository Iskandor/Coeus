#pragma once
#include "NeuralNetwork.h"
#include "ICostFunction.h"
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) RMSProp: public GradientAlgorithm
	{
	public:
		explicit RMSProp(NeuralNetwork* p_network);
		~RMSProp();

		void init(ICostFunction* p_cost_function, float p_alpha, float p_decay = 0.9, float p_epsilon = 1e-8);
	};
}

