#pragma once

namespace Coeus
{
	class ActivationFunctions
	{
	public:
		static double linear(double p_x);
		static double binary(double p_x);
		static double sigmoid(double p_x);
		static double tanh(double p_x);
		static double exponential(double p_x);
		static double softplus(double p_x);
		static double relu(double p_x);
		static double kexponential(double p_x);
		static double gauss(double p_x);
	};
}

