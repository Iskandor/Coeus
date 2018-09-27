#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) SoftplusActivation : public IActivationFunction
	{
	public:
		SoftplusActivation();
		~SoftplusActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;
	};
}

