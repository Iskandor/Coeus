#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) CoreLayer : public BaseLayer
{
public:
	CoreLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation, NeuralGroup* p_parent);
	~CoreLayer();

	void activate(Tensor* p_input);
	void override_params(BaseLayer* p_source);

private:
	Connection *_in_connection;
};

}

