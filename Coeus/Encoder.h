#pragma once
#include "Tensor.h"

using namespace FLAB;

namespace Coeus {

	class __declspec(dllexport) Encoder
	{
	public:
		Encoder();
		~Encoder();

		static void one_hot(int* p_result, int p_size, int p_value);
		static void one_hot(Tensor& p_result, int p_value);
		static void pop_code(Tensor& p_result, double p_value, double p_lower_limit = 0, double p_upper_limit = 1);
		static void grey_code(Tensor& p_result, Tensor& p_bin);

	private:
		static int xor_c(const int a, const int b) { return a == b ? 0 : 1; }
		static int flip(const int c) { return c == 0 ? 1 : 0; }
	};
}
