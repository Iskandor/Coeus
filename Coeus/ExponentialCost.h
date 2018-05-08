#pragma once
#include "ICostFunction.h"

namespace Coeus {
	class __declspec(dllexport) ExponentialCost : public ICostFunction
	{
	public:
		ExponentialCost(double p_tau);
		~ExponentialCost();

		double cost(Tensor* p_prediction, Tensor* p_target) override;
		Tensor cost_deriv(Tensor* p_prediction, Tensor* p_target) override;

	private:
		double _tau;
	};

}


