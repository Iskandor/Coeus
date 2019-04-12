#pragma once
#include "TensorOperatorMKL.h"

namespace Coeus
{
	class __declspec(dllexport) TensorOperator
	{
	public:
		static ITensorOperator& instance() {
			static TensorOperatorMKL op;
			return op;
		}

	private:
		TensorOperator() = default;
		~TensorOperator() = default;
	};
}