#pragma once
#include "Tensor.h"

using namespace FLAB;

namespace Coeus {

	class __declspec(dllexport) Encoder
	{
	public:
		Encoder();
		~Encoder();

		static void one_hot(Tensor& p_result, int p_value);
		static void pop_code(Tensor& p_result, double p_value, double p_lower_limit = 0, double p_upper_limit = 1);
	};
}
