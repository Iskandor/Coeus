#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) BinaryActivation : public IActivationFunction
	{
	public:
		BinaryActivation();
		~BinaryActivation();
		Tensor derivative(Tensor& p_input) override;
		Tensor* backward(Tensor* p_input) override;
		Tensor* forward(Tensor* p_input) override;
	};
}

