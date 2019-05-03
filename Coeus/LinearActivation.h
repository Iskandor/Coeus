#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) LinearActivation : public IActivationFunction
	{
	public:
		LinearActivation();
		~LinearActivation();

		Tensor* forward(Tensor* p_input) override;
		Tensor derivative(Tensor& p_input) override;
		Tensor* backward(Tensor* p_input) override;
	};
}


