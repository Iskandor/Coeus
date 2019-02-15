#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) ReluActivation : public IActivationFunction
	{
	public:
		ReluActivation();
		~ReluActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor derivative(Tensor& p_input) override;
		inline float activate(float p_value) override;
	};
}

