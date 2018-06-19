#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) RecurrentLayer : 	public BaseLayer
{
public:
	RecurrentLayer(string p_id, int p_dim, NeuralGroup::ACTIVATION p_activation);
	~RecurrentLayer();

	void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
	void activate(Tensor* p_input = nullptr) override;
	void override_params(BaseLayer* p_source) override;
	void post_connection(BaseLayer* p_input) override;

private:
	Connection*		_rec_connection;
	NeuralGroup*	_context_group;

};

}

