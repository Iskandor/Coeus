#pragma once

namespace Coeus
{
	class __declspec(dllexport) ActivationFunctions
	{
	public:
		static float linear(float p_x);
		static float binary(float p_x);
		static float sigmoid(float p_x);
		static float tanh(float p_x);
		static float exponential(float p_x);
		static float softplus(float p_x);
		static float relu(float p_x);
		static float kexponential(float p_x);
		static float gauss(float p_x);
	};
}

