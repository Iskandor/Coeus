#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) GaussianActivation : public IActivationFunction
	{
	public:
		GaussianActivation(double p_sigma);
		~GaussianActivation();
		Tensor activate(Tensor& p_input) override;
		Tensor deriv(Tensor& p_input) override;

	private:
		double _sigma;
	};
}


