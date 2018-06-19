#pragma once
#include "Tensor.h"

using namespace FLAB;

namespace Coeus
{
	class __declspec(dllexport) LayerState
	{
	public:
		explicit LayerState(int p_dim);
		virtual ~LayerState();

		Tensor delta;
	};
}


