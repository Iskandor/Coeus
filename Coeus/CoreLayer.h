#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) CoreLayer : public BaseLayer
{
public:
	CoreLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation);
	~CoreLayer();

	void activate(Tensor* p_input, Tensor* p_weights = nullptr);
	void override_params(BaseLayer* p_source);
};

}

