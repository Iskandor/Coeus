#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) ReluActivation : public IActivationFunction
	{
	public:
		ReluActivation();
		~ReluActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;
	};
}

