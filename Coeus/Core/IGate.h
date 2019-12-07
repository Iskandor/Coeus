#pragma once
#include "Tensor.h"

namespace Coeus
{
	class __declspec(dllexport) IGate
	{
	public:
		IGate()
		= default;

		virtual ~IGate()
		= default;

		virtual Tensor*	forward(Tensor* p_input) = 0;
		virtual Tensor*	backward(Tensor* p_input, Tensor* p_x) = 0;
	};
}


