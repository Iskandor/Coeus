#pragma once
#include "ICostFunction.h"

namespace Coeus {
	class __declspec(dllexport) ExponentialCost : public ICostFunction
	{
	public:
		ExponentialCost(float p_tau);
		~ExponentialCost();

		float cost(Tensor* p_prediction, Tensor* p_target) override;
		Tensor cost_deriv(Tensor* p_prediction, Tensor* p_target) override;

	private:
		float _tau;
	};

}


