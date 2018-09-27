#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) SigmoidActivation : public IActivationFunction
	{
	public:
		SigmoidActivation();
		~SigmoidActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;
	};
}


