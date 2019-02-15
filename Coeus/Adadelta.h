#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) Adadelta : public GradientAlgorithm
	{
	public:
		explicit Adadelta(NeuralNetwork* p_network);
		~Adadelta();

		void init(ICostFunction* p_cost_function, float p_alpha = 1, float p_decay = 0.9, float p_epsilon = 1e-8);
	};
}

