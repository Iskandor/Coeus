#include "LayerState.h"

using namespace Coeus;

LayerState::LayerState(int p_dim)
{
	delta = Tensor::Zero({ p_dim });
}

LayerState::~LayerState()
= default;
