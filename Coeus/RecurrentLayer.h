#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) RecurrentLayer : 	public BaseLayer
{
public:
	RecurrentLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation, NeuralGroup* p_parent);
	~RecurrentLayer();

	void activate(Tensor* p_input);
	void override_params(BaseLayer* p_source);

private:
	Connection*		_in_connection;
	Connection*		_rec_connection;
	NeuralGroup*	_context_group;

};

}

