#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) Adagrad : public BaseGradientAlgorithm
	{
	public:
		explicit Adagrad(NeuralNetwork* p_network);
		~Adagrad();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_epsilon = 1e-8);

	private:
		void calc_update() override;
		void init_structures() override;

		double _epsilon;
		map<string, Tensor> _G;
	};

}

