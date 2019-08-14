#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) SoftplusActivation : public IActivationFunction
	{
	public:
		SoftplusActivation();
		~SoftplusActivation();

		Tensor derivative(Tensor& p_input) override;
		Tensor* backward(Tensor* p_input, Tensor* p_x = nullptr) override;
		Tensor* forward(Tensor* p_input) override;
	};
}

