#include "LSTMLayerState.h"

using namespace Coeus;

LSTMLayerState::LSTMLayerState(int p_dim) : LayerState(p_dim)
{
	dc_next = Tensor::Zero({ p_dim });
	dh_next = Tensor::Zero({ p_dim });
}

LSTMLayerState::~LSTMLayerState()
{
}
