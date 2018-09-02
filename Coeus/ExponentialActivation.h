#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) ExponentialActivation : public IActivationFunction
	{
	public:
		explicit ExponentialActivation(int p_k = 1);
		~ExponentialActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;

	private:
		int _k;
	};
}

