#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) TanhActivation : public IActivationFunction
	{
	public:
		TanhActivation();
		~TanhActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;
	};
}

