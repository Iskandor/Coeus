#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) SoftmaxActivation : public IActivationFunction
	{
	public:
		SoftmaxActivation();
		~SoftmaxActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;
	};
}

