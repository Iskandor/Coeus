#pragma once
#include "GradientAlgorithm.h"

namespace Coeus {

	class __declspec(dllexport) ADAM : public GradientAlgorithm
	{
	public:
		explicit ADAM(NeuralNetwork* p_network);
		~ADAM();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_beta1 = 0.9, double p_beta2 = 0.999, double p_epsilon = 1e-8);

		double train(Tensor* p_input, Tensor* p_target) override;
		double train(vector<Tensor*>* p_input, vector<Tensor*>* p_target, int p_batch) override;

	private:
		int _t;
	};
}

