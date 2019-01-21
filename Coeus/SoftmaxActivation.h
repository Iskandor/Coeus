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
		Tensor derivative(Tensor& p_input) override;
		inline double activate(double p_value) override;
	};
}

