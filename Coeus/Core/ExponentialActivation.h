#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) ExponentialActivation : public IActivationFunction
	{
	public:
		explicit ExponentialActivation(int p_k = 1);
		~ExponentialActivation();
		Tensor* backward(Tensor* p_input, Tensor* p_x = nullptr) override;
		Tensor* forward(Tensor* p_input) override;
		Tensor derivative(Tensor& p_input) override;

		json get_json() override;

	private:
		int _k;
	};
}

