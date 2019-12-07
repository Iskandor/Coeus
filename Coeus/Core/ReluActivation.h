#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) ReluActivation : public IActivationFunction
	{
	public:
		ReluActivation();
		~ReluActivation();
		Tensor* backward(Tensor* p_input, Tensor* p_x = nullptr) override;
		Tensor* forward(Tensor* p_input) override;
		Tensor derivative(Tensor& p_input) override;
	};
}
