#pragma once
#include "Tensor.h"

using namespace FLAB;

namespace Coeus {
	class __declspec(dllexport) IActivationFunction
	{
	public:
		IActivationFunction();
		virtual ~IActivationFunction();

		virtual Tensor activate(Tensor& p_input) = 0;
		virtual Tensor deriv(Tensor& p_input) = 0;
	};
}

