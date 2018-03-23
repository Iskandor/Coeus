#pragma once
namespace Coeus
{
	class __declspec(dllexport) ActivationFunctionsDeriv
	{
	public:
		static double dlinear(double p_x);
		static double dbinary(double p_x);
		static double dsigmoid(double p_x);
		static double dtanh(double p_x);
		static double dexponential(double p_x);
		static double dsoftplus(double p_x);
		static double drelu(double p_x);
		static double dkexponential(double p_x);
		static double dgauss(double p_x);
	};
}

