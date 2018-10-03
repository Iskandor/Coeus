#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) RecurrentLayer : 	public BaseLayer
{
public:
	RecurrentLayer(string p_id, int p_dim, ACTIVATION p_activation);
	RecurrentLayer(RecurrentLayer &p_copy);
	~RecurrentLayer();

	void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
	void activate(Tensor* p_input = nullptr) override;
	void override(BaseLayer* p_source) override;

private:
	Connection*		_rec_connection;
	SimpleCellGroup*	_context_group;

};

}

