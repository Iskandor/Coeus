#pragma once
#include "BaseLayer.h"

namespace Coeus {

class __declspec(dllexport) InputLayer : 	public BaseLayer
{
public:
	InputLayer(string p_id, int p_input_dim);
	~InputLayer();

	void activate(Tensor* p_input);
	void override_params(BaseLayer* p_source);
};

}

