#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) TanhActivation : public IActivationFunction
	{
	public:
		TanhActivation();
		~TanhActivation();

		Tensor* backward(Tensor* p_input) override;
		Tensor* forward(Tensor* p_input) override;
		Tensor derivative(Tensor& p_input) override;
	};
}

