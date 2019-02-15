#pragma once

#include "Tensor.h"

using namespace FLAB;

namespace Coeus {

	class __declspec(dllexport) ICostFunction
	{
	public:
		ICostFunction();
		virtual ~ICostFunction();

		virtual float cost(Tensor* p_prediction, Tensor* p_target) = 0;
		virtual Tensor cost_deriv(Tensor* p_prediction, Tensor* p_target) = 0;
	};

}