#pragma once
#include "BaseGradientAlgorithm.h"

namespace Coeus
{
	class __declspec(dllexport) AMSGrad : public BaseGradientAlgorithm
	{
	public:
		explicit AMSGrad(NeuralNetwork* p_network);
		~AMSGrad();

		void init(ICostFunction* p_cost_function, double p_alpha, double p_beta1 = 0.9, double p_beta2 = 0.999, double p_epsilon = 1e-8);

	private:
		void update_momentum(string p_id, Tensor &p_gradient);
		void calc_update() override;
		void init_structures() override;

		double _beta1;
		double _beta2;
		double _epsilon;

		map<string, Tensor> _m;
		map<string, Tensor> _v;
		map<string, Tensor> _v_mean;
	};
}


