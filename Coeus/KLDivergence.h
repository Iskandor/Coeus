#pragma once
#include "ICostFunction.h"

namespace Coeus {

	class __declspec(dllexport) KLDivergence : public ICostFunction
	{
	public:
		KLDivergence();
		~KLDivergence();

		double cost(Tensor* p_prediction, Tensor* p_target) override;
		Tensor cost_deriv(Tensor* p_prediction, Tensor* p_target) override;
	};

}