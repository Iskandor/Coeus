#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) GaussianActivation : public IActivationFunction
	{
	public:
		GaussianActivation(float p_sigma);
		~GaussianActivation();
		Tensor* backward(Tensor* p_input, Tensor* p_x = nullptr) override;
		Tensor* forward(Tensor* p_input) override;
		Tensor derivative(Tensor& p_input) override;

		json get_json() override;
	private:
		float _sigma;
	};
}


