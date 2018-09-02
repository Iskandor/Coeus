#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) BinaryActivation : public IActivationFunction
	{
	public:
		BinaryActivation();
		~BinaryActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;
	};
}

