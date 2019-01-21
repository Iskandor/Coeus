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
		Tensor derivative(Tensor& p_input) override;

		json get_json() override;
	private:
		double _sigma;
	};
}


