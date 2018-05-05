#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) CoreLayer : public BaseLayer
{
	friend class CoreLayerGradient;
public:
	CoreLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation);
	~CoreLayer();

	void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
	void activate(Tensor* p_input = nullptr) override;
	void override_params(BaseLayer* p_source) override;
};

}

