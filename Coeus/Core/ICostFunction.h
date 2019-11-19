#pragma once

#include "Tensor.h"

namespace Coeus {

	class __declspec(dllexport) ICostFunction
	{
	public:
		ICostFunction() = default;
		virtual ~ICostFunction() = default;

		virtual float cost(Tensor* p_prediction, Tensor* p_target) = 0;
		virtual Tensor cost_deriv(Tensor* p_prediction, Tensor* p_target) = 0;
	};

}