#pragma once
#include "BaseLayer.h"
#include "Connection.h"

namespace Coeus {

class __declspec(dllexport) RecurrentLayer : 	public BaseLayer
{
public:
	RecurrentLayer(const string& p_id, int p_dim, ACTIVATION p_activation);
	RecurrentLayer(RecurrentLayer &p_copy);
	~RecurrentLayer();

	void integrate(Tensor* p_input, Tensor* p_weights = nullptr) override;
	void activate(Tensor* p_input = nullptr) override;
	void override(BaseLayer* p_source) override;

private:
	SimpleCellGroup*	_group{};
	SimpleCellGroup*	_context_group;
	Connection*			_rec_connection;
	

};

}

