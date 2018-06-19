#pragma once
#include "BaseLayer.h"

namespace Coeus {

class __declspec(dllexport) InputLayer : 	public BaseLayer
{
public:
	InputLayer(string p_id, int p_input_dim);
	~InputLayer();

	void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
	void activate(Tensor* p_input = nullptr) override;
	void override_params(BaseLayer* p_source) override;
	void post_connection(BaseLayer* p_input) override;
};

}

