#pragma once
#include "IActivationFunction.h"

namespace Coeus
{
	class __declspec(dllexport) ActivationFunctionFactory
	{
	public:
		static IActivationFunction* create_function(ACTIVATION p_type);

	private:
		ActivationFunctionFactory();
		~ActivationFunctionFactory();
	};
}