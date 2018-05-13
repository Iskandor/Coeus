#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus {
	class __declspec(dllexport) AdaMax : public BaseGradientAlgorithm
	{
	public:
		explicit AdaMax(NeuralNetwork* p_network);
		~AdaMax();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_beta1 = 0.9, double p_beta2 = 0.999, double p_epsilon = 1e-8);

	private:
		void update_momentum(string p_id, Tensor &p_gradient);
		void calc_update() override;

		double _beta1;
		double _beta2;
		double _epsilon;

		map<string, Tensor> _momentum1;
		map<string, Tensor> _momentum1_est;
		map<string, Tensor> _inf_norm;
	};
}
