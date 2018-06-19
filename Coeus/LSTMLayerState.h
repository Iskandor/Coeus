#pragma once
#include "LayerState.h"

namespace Coeus
{
	class __declspec(dllexport) LSTMLayerState : public LayerState
	{
	public:
		explicit LSTMLayerState(int p_dim);
		virtual ~LSTMLayerState();

		Tensor dh_next;
		Tensor dc_next;
	};
}