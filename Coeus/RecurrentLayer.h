#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) RecurrentLayer : 	public BaseLayer
{
public:
	RecurrentLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation);
	~RecurrentLayer();

	void activate(Tensor* p_input, Tensor* p_weights = nullptr);
	void override_params(BaseLayer* p_source);

private:
	Connection*		_rec_connection;
	NeuralGroup*	_context_group;

};

}

