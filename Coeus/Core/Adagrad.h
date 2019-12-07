#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) Adagrad : public GradientAlgorithm
	{
	public:
		explicit Adagrad(NeuralNetwork* p_network);
		~Adagrad();

		void init(ICostFunction* p_cost_function, float p_alpha, float p_epsilon = 1e-8);
	};

}

