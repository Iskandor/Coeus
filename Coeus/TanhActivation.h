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
		Tensor derivative(Tensor& p_input) override;
		inline double activate(double p_value) override;
	};
}

